from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import FIFAField, HLAdultField, CustomField36x55
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger()


# ==================== 守门员状态枚举 ====================
class GoalkeeperState:
    """守门员状态枚举"""
    IDLE = "idle"              # 待命
    POSITIONING = "positioning"  # 站位调整
    DIVING = "diving"          # 扑救
    RUSHING = "rushing"        # 出击
# =====================================================


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
        
        # ==================== 守门员状态初始化 ====================
        self.goalkeeper_state = GoalkeeperState.IDLE
        self.last_ball_position = None  
        # =========================================================

    # ==================== 新增：简化版守门员防守 ====================
    def goalkeeper_defend_simple(self) -> None:
        
        field = self.agent.world.field
        field_length = field.get_length()
        
        ball_pos = self.agent.world.ball_pos
        my_pos = self.agent.world.global_position
        
        # ===== 1. 球门参数 =====
        # 本方球门线位置
        goal_line_z = -field_length / 2 + 1.0
        
        # 球门宽度
        goal_width = 5.0
        
        # ===== 2. 计算守门员目标位置 =====
        # 根据球的 x 坐标在球门线上移动
        # 球在左边，守门员向左；球在右边，守门员向右
        target_x = np.clip(ball_pos[0] * 0.5, -goal_width/2 * 0.7, goal_width/2 * 0.7)
        
        # 默认站在球门线上
        target_z = goal_line_z
        
        # 如果球靠近球门，守门员出击
        # 球在 z > -field_length/2 + 8 时，认为球进入危险区域
        if ball_pos[1] > -field_length / 2 + 8:
            target_z = goal_line_z + 1.0
        
        # 球在非常危险的位置 (z > -field_length/2 + 3)，快速出击
        if ball_pos[1] > -field_length / 2 + 3:
            target_z = goal_line_z + 2.0
        
        target_pos = np.array([target_x, target_z])
        
        # ===== 3. 守门员面向球 =====
        ball_direction = ball_pos[:2] - my_pos[:2]
        ball_distance = np.linalg.norm(ball_direction)
        
        if ball_distance > 0.01:
            target_angle = MathOps.vector_angle(ball_direction / ball_distance)
        else:
            target_angle = 0.0
        
        # ===== 4. 移动到目标位置 =====
        distance_to_target = np.linalg.norm(my_pos[:2] - target_pos)
        
        if distance_to_target > 0.05:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_pos,
                is_target_absolute=True,
                orientation=target_angle
            )
        else:
            self.agent.skills_manager.execute("Neutral")
    # ==================== 简化版守门员结束 ====================

    # ==================== 带预测的守门员 ====================
    def goalkeeper_defend_with_prediction(self) -> None:
        """
        守门员防守 + 简单球路预测（通过位置差估算速度）
        比简化版更智能，但依赖帧率稳定性
        """
        field = self.agent.world.field
        field_length = field.get_length()
        
        ball_pos = self.agent.world.ball_pos
        my_pos = self.agent.world.global_position
        
        goal_line_z = -field_length / 2 + 1.0
        goal_width = 5.0
        
        current_ball_pos = ball_pos[:2].copy()
        
        # ===== 1. 估算球速（通过位置差） =====
        estimated_speed = 0.0
        predicted_pos = current_ball_pos.copy()
        
        if self.last_ball_position is not None:
            # 计算位移
            displacement = current_ball_pos - self.last_ball_position
            distance = np.linalg.norm(displacement)
            
            # 估算速度（假设 20ms 一帧）
            estimated_speed = distance / 0.02
            
            # 如果球移动较快，预测落点
            if estimated_speed > 2.0:
                # 线性外推预测
                predicted_pos = current_ball_pos + displacement * 5
        
        # 保存当前位置供下一帧使用
        self.last_ball_position = current_ball_pos
        
        # ===== 2. 计算目标位置 =====
        # 使用预测位置来决定守门员站位
        target_x = np.clip(predicted_pos[0] * 0.5, -goal_width/2 * 0.7, goal_width/2 * 0.7)
        target_z = goal_line_z
        
        # 球靠近时出击
        if ball_pos[1] > -field_length / 2 + 8:
            target_z = goal_line_z + 1.0
        if ball_pos[1] > -field_length / 2 + 3:
            target_z = goal_line_z + 2.0
        
        # 如果预测到球速很快，提前移动
        if estimated_speed > 3.0:
            target_x = np.clip(predicted_pos[0] * 0.6, -goal_width/2 * 0.8, goal_width/2 * 0.8)
        
        target_pos = np.array([target_x, target_z])
        
        # ===== 3. 面向球并移动 =====
        ball_direction = ball_pos[:2] - my_pos[:2]
        if np.linalg.norm(ball_direction) > 0.01:
            target_angle = MathOps.vector_angle(ball_direction / np.linalg.norm(ball_direction))
        else:
            target_angle = 0.0
        
        distance_to_target = np.linalg.norm(my_pos[:2] - target_pos)
        
        if distance_to_target > 0.05:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target_pos,
                is_target_absolute=True,
                orientation=target_angle
            )
        else:
            self.agent.skills_manager.execute("Neutral")
    # ==================== 带预测的守门员结束 ====================

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

        # ==================== 守门员使用简化版 ====================
        # Goalkeeper (player 1) 执行防守
        elif self.agent.world.number == 1:
            # 选择其中一种：
            self.goalkeeper_defend_simple()  # 简化版（推荐）
            # self.goalkeeper_defend_with_prediction()  # 带预测版（可选）
        
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