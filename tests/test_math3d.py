"""Rotation composition, quaternion algebra, and BVH→Roblox axis conversion."""

import math
import unittest

from mocap2r15.math3d import (
    VEC_X,
    VEC_Y,
    VEC_Z,
    almost_vec,
    conjugate,
    euler_matrix,
    euler_to_quat,
    forward_axis_quaternion,
    mat_vec,
    quat_angle,
    quat_between_angle,
    quat_from_axis_angle,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_slerp,
    quat_to_mat,
    mat_to_quat,
    yaw_about_y,
)


class EulerOrderTests(unittest.TestCase):
    def test_z_90_sends_x_to_y(self):
        # A single Z channel is a pure Rz, independent of the other axes.
        m = euler_matrix("ZXY", (90.0, 0.0, 0.0))
        self.assertTrue(almost_vec(mat_vec(m, VEC_X), (0.0, 1.0, 0.0), 1e-6))

    def test_zyx_matches_blender_bvh_importer(self):
        # Importer: Euler((x, y, z), 'ZYX'[::-1]) = Euler((90°, 0, 90°), 'XYZ')
        # = Rz(90) @ Rx(90). +X goes to +Y. The reversed composition
        # Rx @ Rz sent +X to +Z and is what stuck CMU arms behind the body.
        m = euler_matrix("ZYX", (90.0, 0.0, 90.0))
        self.assertTrue(almost_vec(mat_vec(m, VEC_X), (0.0, 1.0, 0.0), 1e-6))
        self.assertTrue(almost_vec(mat_vec(m, VEC_Y), (0.0, 0.0, 1.0), 1e-6))
        self.assertTrue(almost_vec(mat_vec(m, VEC_Z), (1.0, 0.0, 0.0), 1e-6))

    def test_intrinsic_x_bends_down_leg_backward(self):
        # Positive Rx on a bone that points down (-Y) tilts it toward -Z.
        m = euler_matrix("ZYX", (0.0, 0.0, 30.0))
        bent = mat_vec(m, (0.0, -1.0, 0.0))
        self.assertLess(bent[2], -0.4)
        self.assertLess(bent[1], -0.8)

    def test_composition_of_two_z_90_is_180(self):
        a = euler_to_quat("Z", (90.0,))
        b = euler_to_quat("Z", (90.0,))
        # Parent rotation is applied first, so it sits on the right: q = b * a
        # only if b is the child. Here both are the same joint applied twice:
        # R(q_mul(b, a)) = Rb @ Ra, so the vector sees a first, then b.
        both = quat_mul(b, a)
        self.assertAlmostEqual(math.degrees(quat_angle(both)), 180.0, places=4)
        self.assertTrue(almost_vec(quat_rotate(both, VEC_X), (-1.0, 0.0, 0.0), 1e-6))


class QuaternionTests(unittest.TestCase):
    def test_matrix_roundtrip(self):
        q = quat_from_axis_angle((0.2, 0.8, -0.3), 1.2)
        back = mat_to_quat(quat_to_mat(q))
        # q and -q are the same rotation.
        if back[0] * q[0] < 0:
            back = tuple(-c for c in back)
        self.assertTrue(almost_vec(back, q, 1e-6))

    def test_mul_applies_right_factor_first(self):
        qx = quat_from_axis_angle(VEC_X, math.pi / 2)
        qy = quat_from_axis_angle(VEC_Y, math.pi / 2)
        # quat_mul(qy, qx) = Ry @ Rx, so Rx happens first: +Z -> -Y -> -Y.
        v = quat_rotate(quat_mul(qy, qx), VEC_Z)
        rx_then = quat_rotate(qx, VEC_Z)
        self.assertTrue(almost_vec(rx_then, (0.0, -1.0, 0.0), 1e-6))
        self.assertTrue(almost_vec(v, quat_rotate(qy, rx_then), 1e-6))

    def test_slerp_midpoint_is_45_degrees(self):
        a = quat_from_axis_angle(VEC_Y, 0.0)
        b = quat_from_axis_angle(VEC_Y, math.pi / 2)
        mid = quat_slerp(a, b, 0.5)
        self.assertAlmostEqual(math.degrees(quat_angle(mid)), 45.0, places=3)

    def test_slerp_takes_shortest_path(self):
        a = quat_from_axis_angle(VEC_Y, 0.0)
        # Long way around stored as the opposite quaternion of a small step.
        b = quat_from_axis_angle(VEC_Y, math.radians(20))
        b_neg = tuple(-c for c in b)
        mid = quat_slerp(a, b_neg, 0.5)
        self.assertLess(math.degrees(quat_angle(mid)), 15.0)

    def test_inverse_cancels(self):
        q = quat_from_axis_angle((1, 2, 3), 0.7)
        self.assertAlmostEqual(quat_angle(quat_mul(q, quat_inv(q))), 0.0, places=5)


class AxisConversionTests(unittest.TestCase):
    def test_plus_z_faces_roblox_forward(self):
        c = forward_axis_quaternion("z")
        self.assertTrue(almost_vec(quat_rotate(c, (0, 0, 1)), (0, 0, -1), 1e-6))
        self.assertTrue(almost_vec(quat_rotate(c, (0, 1, 0)), (0, 1, 0), 1e-6))
        # 180° yaw also flips left/right, which keeps the frame right-handed.
        self.assertTrue(almost_vec(quat_rotate(c, (1, 0, 0)), (-1, 0, 0), 1e-6))

    def test_plus_x_faces_roblox_forward(self):
        c = forward_axis_quaternion("x")
        self.assertTrue(almost_vec(quat_rotate(c, (1, 0, 0)), (0, 0, -1), 1e-6))
        self.assertTrue(almost_vec(quat_rotate(c, (0, 1, 0)), (0, 1, 0), 1e-6))

    def test_minus_z_is_identity(self):
        c = forward_axis_quaternion("-z")
        self.assertAlmostEqual(quat_angle(c), 0.0, places=6)

    def test_conjugation_preserves_yaw_direction(self):
        c = forward_axis_quaternion("z")
        yaw = quat_from_axis_angle(VEC_Y, math.radians(30))
        turned = conjugate(yaw, c)
        # Yaw commutes with a yaw-only axis conversion, so the angle is unchanged.
        self.assertLess(math.degrees(quat_between_angle(yaw, turned)), 1e-3)


class YawSplitTests(unittest.TestCase):
    def test_pure_yaw_has_no_rest(self):
        q = quat_from_axis_angle(VEC_Y, math.radians(40))
        q_yaw, q_rest = yaw_about_y(q)
        self.assertLess(math.degrees(quat_between_angle(q, q_yaw)), 1e-3)
        self.assertLess(math.degrees(quat_angle(q_rest)), 1e-3)

    def test_split_recombines(self):
        q = quat_mul(
            quat_from_axis_angle(VEC_Y, math.radians(25)),
            quat_from_axis_angle(VEC_X, math.radians(15)),
        )
        q_yaw, q_rest = yaw_about_y(q)
        back = quat_mul(q_yaw, q_rest)
        self.assertLess(math.degrees(quat_between_angle(q, back)), 1e-3)


if __name__ == "__main__":
    unittest.main()
