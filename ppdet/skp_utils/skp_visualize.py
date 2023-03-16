"""
@File  : skp_visualize.py
@Author: tao.jing
@Date  : 2022/5/7
@Desc  :
"""

__all__ = [
    'visualize_kps'
]


def visualize_kps(img, joints):
    import matplotlib.pyplot as plt
    print(joints)
    ax = plt.gca()
    ax.imshow(img, alpha=1)
    cmap = ['r', 'b', 'g']
    for idx, kp_coor in enumerate(joints):
        circ = plt.Circle((kp_coor[0], kp_coor[1]), radius=10, color=cmap[idx])
        ax.add_patch(circ)
    plt.show()