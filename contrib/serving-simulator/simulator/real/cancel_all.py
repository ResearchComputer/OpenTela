import paramiko
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLUSTERS = ["bristen", "clariden"]

def cancel_all_jobs():
    for cluster in CLUSTERS:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jump_client = None
        
        try:
            # Load system host keys and user config
            client.load_system_host_keys()
            
            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)
            
            user_config = ssh_config.lookup(cluster)
            
            connect_args = {
                'hostname': user_config.get('hostname', cluster),
                'username': user_config.get('user'),
                'port': int(user_config.get('port', 22)),
                'key_filename': user_config.get('identityfile')
            }
            
            # Filter None values
            connect_args = {k: v for k, v in connect_args.items() if v is not None}
            
            # Check for ProxyJump
            proxy_jump = user_config.get('proxyjump')
            sock = None
            
            if proxy_jump:
                logger.info(f"Found ProxyJump: {proxy_jump}")
                jump_config = ssh_config.lookup(proxy_jump)
                
                jump_client = paramiko.SSHClient()
                jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                jump_client.load_system_host_keys()
                
                jump_args = {
                    'hostname': jump_config.get('hostname', proxy_jump),
                    'username': jump_config.get('user'),
                    'port': int(jump_config.get('port', 22)),
                    'key_filename': jump_config.get('identityfile')
                }
                jump_args = {k: v for k, v in jump_args.items() if v is not None}
                
                logger.info(f"Connecting to jump host {proxy_jump}...")
                jump_client.connect(**jump_args)
                
                dest_addr = (connect_args['hostname'], connect_args['port'])
                sock = jump_client.get_transport().open_channel("direct-tcpip", dest_addr, ('127.0.0.1', 0))

            logger.info(f"Connecting to {cluster} ({connect_args.get('hostname')})...")
            client.connect(**connect_args, sock=sock)
            
            # Execute scancel
            # We need to get the username on the remote host to be safe, or just use $USER if we trust the shell expansion
            # "scancel -u $USER" should work in bash
            command = "scancel -u $USER"
            logger.info(f"Executing on {cluster}: {command}")
            stdin, stdout, stderr = client.exec_command(command)
            
            exit_status = stdout.channel.recv_exit_status()
            out_str = stdout.read().decode().strip()
            err_str = stderr.read().decode().strip()
            
            if exit_status == 0:
                logger.info(f"Successfully cancelled jobs on {cluster}. Output: {out_str}")
            else:
                logger.error(f"Failed to cancel jobs on {cluster} (exit {exit_status}): {err_str}")
                
        except Exception as e:
            logger.error(f"Failed to connect/execute on {cluster}: {e}")
        finally:
            client.close()
            if jump_client:
                jump_client.close()

if __name__ == "__main__":
    cancel_all_jobs()
