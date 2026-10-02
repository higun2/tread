import unittest

import torch

from tread_routesync.loss import dense_sparse_relative_l2_loss


class RelativeL2Tests(unittest.TestCase):
    def test_formula_and_teacher_stop(self):
        student = torch.tensor([[[2., 4.], [1., 3.]]], requires_grad=True)
        teacher = torch.tensor([[[1., 2.], [2., 1.]]], requires_grad=True)
        loss = dense_sparse_relative_l2_loss(student, teacher)
        expected = ((student - teacher.detach()).square().sum(-1)
                    / (teacher.detach().square().sum(-1) + 1e-8)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_identical_zero_and_bf16(self):
        x = torch.randn(2, 3, 4)
        self.assertEqual(dense_sparse_relative_l2_loss(x, x).item(), 0)
        student = torch.ones(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.zeros_like(student)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = dense_sparse_relative_l2_loss(student, teacher)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        torch.testing.assert_close(loss, torch.tensor(4 / 1e-8))

    def test_validation(self):
        x = torch.ones(2, 3, 4)
        for eps in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                dense_sparse_relative_l2_loss(x, x, eps)
        with self.assertRaises(ValueError):
            dense_sparse_relative_l2_loss(x, x[:, :1])


if __name__ == '__main__':
    unittest.main()
