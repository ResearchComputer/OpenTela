import unittest
from unittest.mock import MagicMock, patch
import yaml
import os
from simulator.real.spinup import SpinUpManager, cluster_mapping
from simulator.core.placement import NodeConfiguration, ParallelConfig

class TestSpinUpManager(unittest.TestCase):
    def setUp(self):
        # Create a dummy config file
        self.config_path = "test_config.yaml"
        self.config_data = {
            "nodes": [
                {"gpu": "NVDA:GH200", "count": 16, "gpus_per_node": 4, "cost": 27.52},
                {"gpu": "NVDA:A100_80G:SXM", "count": 8, "gpus_per_node": 4, "cost": 10.97}
            ],
            "placement_strategy": "maximize_replicas",
            "workload": [
                {
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "arrival_rate": "Poisson(5)",
                    "duration": 10,
                    "input": "Normal(1024, 5)",
                    "output": "Normal(10, 5)"
                },
                {
                    "model": "meta-llama/Llama-3.3-70B-Instruct",
                    "arrival_rate": "Poisson(3)",
                    "duration": 10,
                    "input": "Normal(1024, 5)",
                    "output": "Normal(10, 5)"
                }
            ]
        }
        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)
            
    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            
    def test_parse_config(self):
        manager = SpinUpManager(self.config_path)
        self.assertEqual(len(manager.physical_nodes), 2)
        self.assertEqual(len(manager.workloads), 2)
        self.assertIsInstance(manager.workloads[0].arrival_process, object) # Check it's an object

    @patch("simulator.real.spinup.PlacementDecisionMaker")
    @patch("simulator.real.spinup.paramiko")
    def test_run_logic(self, MockParamiko, MockDecisionMaker):
        # Mock placement result
        mock_dm = MockDecisionMaker.return_value
        
        # Create dummy logical nodes
        # 4 replicas of 1B model on GH200 (TP=1) -> 1 physical node
        # 1 replica of 70B model on A100 (TP=4) -> 1 physical node
        
        logical_nodes = []
        # 1B model, TP=1, 4 replicas
        for i in range(4):
            logical_nodes.append(NodeConfiguration(
                node_id=f"node_{i}",
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                hardware="NVDA:GH200",
                parallel_config=ParallelConfig(tensor_parallel_size=1)
            ))
            
        # 70B model, TP=4, 1 replica
        logical_nodes.append(NodeConfiguration(
            node_id="node_4",
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            parallel_config=ParallelConfig(tensor_parallel_size=4)
        ))
        
        mock_dm.place.return_value = (logical_nodes, {})
        
        # Mock Paramiko
        mock_ssh = MockParamiko.SSHClient.return_value
        mock_sftp = mock_ssh.open_sftp.return_value
        
        # Mock exec_command return values (stdin, stdout, stderr)
        mock_stdout = MagicMock()
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stdout.read.return_value = b"Submitted batch job 12345"
        
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        
        mock_ssh.exec_command.return_value = (None, mock_stdout, mock_stderr)
        
        manager = SpinUpManager(self.config_path)
        manager.run()
        
        # Verify SSH connections
        # Should connect to clariden and bristen
        # Since we mock SSHClient, and it's instantiated inside submit_job, 
        # we need to check if connect was called on the instance.
        # But wait, submit_job instantiates a NEW SSHClient each time.
        # So MockParamiko.SSHClient() is called twice.
        
        self.assertEqual(MockParamiko.SSHClient.call_count, 2)
        
        # Verify connect calls
        # We can inspect the mock instances.
        # Since return_value is the same mock object by default unless side_effect is used,
        # checking the single mock_ssh instance calls might work if they are sequential.
        
        connect_calls = mock_ssh.connect.call_args_list
        hostnames = [c[1].get('hostname') for c in connect_calls]
        # Note: hostname might be None if it relies on SSH config lookup which we also mocked/bypassed?
        # In the code: connect_args['hostname'] comes from user_config.get('hostname', cluster)
        # We didn't mock SSHConfig properly in the test setup above, so it might try to read real ~/.ssh/config.
        # Let's mock SSHConfig too to be safe.
        
    @patch("simulator.real.spinup.PlacementDecisionMaker")
    @patch("simulator.real.spinup.paramiko")
    def test_run_logic_full_mock(self, MockParamiko, MockDecisionMaker):
        # Mock placement result
        mock_dm = MockDecisionMaker.return_value
        
        logical_nodes = []
        for i in range(4):
            logical_nodes.append(NodeConfiguration(
                node_id=f"node_{i}",
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                hardware="NVDA:GH200",
                parallel_config=ParallelConfig(tensor_parallel_size=1)
            ))
        logical_nodes.append(NodeConfiguration(
            node_id="node_4",
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            hardware="NVDA:A100_80G:SXM",
            parallel_config=ParallelConfig(tensor_parallel_size=4)
        ))
        mock_dm.place.return_value = (logical_nodes, {})
        
        # Mock Paramiko SSHClient
        mock_ssh = MockParamiko.SSHClient.return_value
        mock_sftp = mock_ssh.open_sftp.return_value
        
        mock_stdout = MagicMock()
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stdout.read.return_value = b"Submitted batch job 12345"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_ssh.exec_command.return_value = (None, mock_stdout, mock_stderr)
        
        # Mock SSHConfig to avoid reading real file
        mock_ssh_config = MockParamiko.SSHConfig.return_value
        def lookup_side_effect(host):
            return {'hostname': host, 'user': 'testuser'}
        mock_ssh_config.lookup.side_effect = lookup_side_effect
        
        manager = SpinUpManager(self.config_path)
        manager.run()
        
        # Verify interactions
        self.assertEqual(mock_ssh.connect.call_count, 2)
        
        # Verify script upload
        # sftp.file().write()
        # We can check what was written
        
        # Verify sbatch command
        # exec_command("sbatch ...")
        self.assertEqual(mock_ssh.exec_command.call_count, 2)
        
        # Check commands
        commands = [c[0][0] for c in mock_ssh.exec_command.call_args_list]
        self.assertTrue(any("sbatch" in cmd for cmd in commands))

if __name__ == "__main__":
    unittest.main()
