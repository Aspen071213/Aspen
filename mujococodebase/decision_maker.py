from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import FIFAField, HLAdultField, CustomField36x55
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger()


# ==================== 新增：守门员状态枚举（放在类外部） ====================
class GoalkeeperState:
    """守门员状态枚举"""
    IDLE = "idle"              # 待命
    POSITIONING = "positioning"  # 站位调整
    DIVING = "diving"          # 扑救
    RUSHING = "rushing"        # 出击
# ========================================================================


class DecisionMaker:
    """
    Responsible for deciding what the agent should do at each moment.

    This class is called every simulation step to update the agent's behavior
    based on the current state of the world and game conditions.
    """

    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] = {
        FIFAField: {
            1: (2.1, 0, 0),
            2: (22.0, 12.0, 0),
            3: (22.0, 4.0, 0),
            4: (22.0, -4.0, 0),
            5: (22.0, -12.0, 0),
            6: (15.0, 0.0, 0),
            7: (4.0, 16.0, 0),
        },
        HLAdultField: {
            1: (7.0, 0.0, 0),
            2: (2.0, -1.5, 0),
            3: (2.0, 1.5, 0),
        },
        # ==================== 修复语法错误 ====================
        CustomField36x55: {
            1: (0.0, -26.0, 0),   # 守门员 
            2: (-10.0, -14.0, 0), # 左后卫  
            3: (0.0, -16.0, 0),   # 中后卫 
            4: (10.0, -14.0, 0),  # 右后卫 
            5: (-8.0, -3.0, 0),   # 左中场 
            6: (0.0, 0.0, 0),     # 中场核心 
            7: (8.0, -3.0, 0),    # 右中场 
        }
    } 

    def __init__(self, agent):
        """
        Creates a new DecisionMaker linked to the given agent.

        Args:
            agent: The main agent that owns this DecisionMaker.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False
        
        # ==================== 新增：守门员状态初始化 ====================
        self.goalkeeper_state = GoalkeeperState.IDLE
        self.last_ball_position = None
        self.ball_position_history = []  # 球的位置历史
        # =============================================================

    # ==================== 新增：守门员防守方法 ====================
    def goalkeeper_defend_advanced(self) -> None:
        """
        高级守门员防守逻辑
        """
        field = self.agent.world.field
        field_width = field.get_width()
        field_length = field.get_length()
        
        ball_pos = self.agent.world.ball_pos
        my_pos = self.agent.world.global_position
        
        # 球门参数
        goal_width = 5.0
        goal_line_z = -field_length / 2 + 1.0
        
        # ===== 1. 预测球的落点 =====
        predicted_ball_pos = self._predict_ball_position()
        
        # ===== 2. 根据球的位置决定守门员行为 =====
        ball_distance_to_goal = abs(ball_pos[1] - goal_line_z)
        ball_speed = np.linalg.norm(self.agent.world.ball_vel[:2]) if hasattr(self.agent.world, 'ball_vel') and self.agent.world.ball_vel is not None else 0
        
        # 情况1: 球在危险区域且速度很快 → 扑救
        if ball_distance_to_goal < 5.0 and ball_speed > 2.0:
            self.goalkeeper_state = GoalkeeperState.DIVING
            self._goalkeeper_dive_advanced(predicted_ball_pos)
            return
        
        # 情况2: 球在远射范围 → 站位封堵角度
        elif ball_distance_to_goal < 15.0:
            self.goalkeeper_state = GoalkeeperState.POSITIONING
            # 计算封堵角度
            target_x = self._calculate_blocking_angle(ball_pos[:2])
            target_pos = np.array([target_x, goal_line_z])
            
            # 移动到位
            self._move_to_position(target_pos, ball_pos[:2])
            return
        
        # 情况3: 球在远处 → 保持中立位置
        else:
            self.goalkeeper_state = GoalkeeperState.IDLE
            # 回到球门中央
            target_pos = np.array([0.0, goal_line_z])
            self._move_to_position(target_pos, ball_pos[:2])

    def _predict_ball_position(self) -> np.ndarray:
        """
        预测球的落点
        
        Returns:
            np.ndarray: 预测的球位置 [x, z]
        """
        ball_pos = self.agent.world.ball_pos
        ball_vel = self.agent.world.ball_vel if hasattr(self.agent.world, 'ball_vel') else None
        
        if ball_vel is None or np.linalg.norm(ball_vel) < 0.01:
            return ball_pos[:2]
        
        # 预测0.5秒后的位置（简化版）
        prediction_time = 0.5
        predicted_pos = ball_pos[:2] + ball_vel[:2] * prediction_time
        
        # 限制在球场范围内
        field = self.agent.world.field
        half_width = field.get_width() / 2
        half_length = field.get_length() / 2
        
        predicted_pos[0] = np.clip(predicted_pos[0], -half_width, half_width)
        predicted_pos[1] = np.clip(predicted_pos[1], -half_length, half_length)
        
        return predicted_pos

    def _calculate_blocking_angle(self, ball_pos: np.ndarray) -> float:
        """
        计算守门员应该站在球门线的哪个位置来封堵射门角度
        
        Args:
            ball_pos: 球的位置 [x, z]
            
        Returns:
            float: 守门员应该在球门线上的x位置
        """
        goal_width = 5.0
        goal_center_x = 0.0
        goal_line_z = -self.agent.world.field.get_length() / 2 + 1.0
        
        # 简单方法：守门员站在球和球门中心的连线上
        ball_to_goal = np.array([goal_center_x, goal_line_z]) - ball_pos
        distance = np.linalg.norm(ball_to_goal)
        
        target_x = 0.0
        if distance > 0.01:
            # 球和球门中心的连线
            direction = ball_to_goal / distance
            # 守门员在球门线上
            target_x = goal_center_x + direction[0] * 0.8
        
        # 限制在球门范围内
        target_x = np.clip(target_x, -goal_width/2 * 0.7, goal_width/2 * 0.7)
        
        return target_x

    def _move_to_position(self, target_pos: np.ndarray, face_pos: np.ndarray) -> None:
        """
        移动到指定位置并面向指定点
        
        Args:
            target_pos: 目标位置 [x, z]
            face_pos: 面向的位置 [x, z]
        """
        my_pos = self.agent.world.global_position[:2]
        
        # 计算朝向角度
        face_direction = face_pos - my_pos
        distance = np.linalg.norm(face_direction)
        
        if distance > 0.01:
            face_angle = MathOps.vector_angle(face_direction / distance)
        else:
            face_angle = 0.0
        
        # 移动到目标位置
        distance_to_target = np.linalg.norm(my_pos - target_pos)
        
        if distance_to_target > 0.05:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_pos,
                is_target_absolute=True,
                orientation=face_angle
            )
        else:
            self.agent.skills_manager.execute("Neutral")

    def _goalkeeper_dive_advanced(self, target_pos: np.ndarray) -> None:
        """
        高级扑救动作
        
        Args:
            target_pos: 扑救目标位置
        """
        my_pos = self.agent.world.global_position[:2]
        
        # 扑救速度（快速移动）
        dive_speed = 3.0
        
        # 如果距离远，使用快速移动
        distance = np.linalg.norm(my_pos - target_pos)
        
        if distance > 0.1:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_pos,
                is_target_absolute=True,
                speed=dive_speed
            )
        else:
            self.agent.skills_manager.execute("Neutral")
        
        # 扑救完成后回到待命状态
        if distance < 0.1:
            self.goalkeeper_state = GoalkeeperState.IDLE
    # ==================== 守门员方法结束 ====================

    def update_current_behavior(self) -> None:
        """
        Chooses what the agent should do in the current step.

        This function checks the game state and decides which behavior
        or skill should be executed next.
        """

        if self.agent.world.playmode is PlayModeEnum.GAME_OVER:
            return

        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2],
                rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
            )

        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")

        # ==================== 修改：守门员使用高级防守逻辑 ====================
        # Goalkeeper (player 1) 执行防守
        elif self.agent.world.number == 1:
            self.goalkeeper_defend_advanced()  # 使用高级守门员防守
        
        # ==================== 其他球员逻辑 ====================
        elif self.agent.world.playmode is PlayModeEnum.PLAY_ON:
            self.carry_ball()
        elif self.agent.world.playmode in (PlayModeEnum.BEFORE_KICK_OFF, PlayModeEnum.THEIR_GOAL, PlayModeEnum.OUR_GOAL):
            self.agent.skills_manager.execute("Neutral")
        else:
            self.carry_ball()

        self.agent.robot.commit_motor_targets_pd()

    def carry_ball(self):
        """
        Basic example of a behavior: moves the robot toward the goal while handling the ball.
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return 
        ball_to_goal_dir = ball_to_goal / bg_norm

        dist_from_ball_to_start_carrying = 0.30
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if not aligned or not behind_ball:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=None if np.linalg.norm(my_pos - carry_ball_pos) > 2 else desired_orientation
            )
        else:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )