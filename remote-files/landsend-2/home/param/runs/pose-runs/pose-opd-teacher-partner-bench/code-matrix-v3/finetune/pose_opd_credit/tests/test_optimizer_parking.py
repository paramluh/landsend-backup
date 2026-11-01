import unittest

import torch

from finetune.pose_opd_credit.optimizer_parking import OptimizerStateParking


class OptimizerParkingTests(unittest.TestCase):
    def test_empty_optimizer_is_guarded_and_restored_on_exception(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter])
        parking = OptimizerStateParking(optimizer)
        with self.assertRaisesRegex(ValueError, 'rollout failed'):
            with parking:
                self.assertTrue(parking.parked)
                self.assertEqual(parking.stats['parked_tensors'], 0)
                for operation in (optimizer.step, optimizer.state_dict,
                                  lambda: optimizer.load_state_dict({}), parking.park,
                                  OptimizerStateParking(optimizer).park):
                    with self.assertRaisesRegex(RuntimeError, 'parked'):
                        operation()
                raise ValueError('rollout failed')
        self.assertFalse(parking.parked)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(optimizer.state[parameter]['step'], 1)
        self.assertIn('state', optimizer.state_dict())

    def test_cpu_state_and_partially_initialized_optimizer_remain_exact(self):
        parameters = [torch.nn.Parameter(torch.tensor([1., -2.])),
                      torch.nn.Parameter(torch.tensor([3., -4.]))]
        optimizer = torch.optim.AdamW(parameters, lr=3e-6)
        parameters[0].grad = torch.tensor([.25, -.5])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        previous = dict(optimizer.state[parameters[0]])
        with OptimizerStateParking(optimizer):
            self.assertNotIn(parameters[1], optimizer.state)
            for key, value in previous.items():
                self.assertIs(optimizer.state[parameters[0]][key], value)
        for key, value in previous.items():
            self.assertIs(optimizer.state[parameters[0]][key], value)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(optimizer.state[parameters[0]]['step'], 2)
        self.assertEqual(optimizer.state[parameters[1]]['step'], 1)

    def test_failed_park_removes_guards(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter])
        optimizer.state[parameter]['unsupported'] = {'nested': torch.zeros(2)}
        parking = OptimizerStateParking(optimizer)
        with self.assertRaisesRegex(TypeError, 'nested Adam state'):
            parking.park()
        self.assertFalse(parking.parked)
        optimizer.state.clear()
        self.assertIn('state', optimizer.state_dict())
        optimizer.step()

    def test_non_adam_optimizer_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'Adam'):
            OptimizerStateParking(torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1))


if __name__ == '__main__':
    unittest.main()
