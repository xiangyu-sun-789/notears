# import numpy as np
# from lingam.utils import make_dot
#
# if __name__ == "__main__":
#     # draw DAG
#     variable_names = ['xAdjCoord_1', 'yAdjCoord_1', 'scoreDifferential_1',
#                       'manpowerSituation_1', 'outcome_1', 'velocity_x_1', 'velocity_y_1',
#                       'time_remain_1', 'duration_1', 'angle2gate_1', 'xAdjCoord_2',
#                       'yAdjCoord_2', 'scoreDifferential_2', 'manpowerSituation_2',
#                       'outcome_2', 'velocity_x_2', 'velocity_y_2', 'time_remain_2',
#                       'duration_2', 'home_or_away', 'angle2gate_2', 'shot_1', 'shot_2',
#                       'reward_2']
#
#     adjacency_matrix_ = np.loadtxt(open("./notears_DAGs_0.0_prior_knowledge.csv", "rb"), delimiter=",", skiprows=0)
#
#     # direction of the adjacency matrix needs to be transposed.
#     # in LINGAM, the adjacency matrix is defined as column variable -> row variable
#     # in NOTEARS, the W is defined as row variable -> column variable
#     dot = make_dot(np.transpose(adjacency_matrix_), labels=variable_names)
#
#     dot.format = 'png'
#     dot.render('./DAG_0.0_prior_knowledge_image')
