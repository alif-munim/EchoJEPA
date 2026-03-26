#!/bin/bash
# Add this to the end of on_create.sh (or call it from there) to enable SSM access.
# After updating, upload to s3://sagemaker-echojepa-h100-march-0d224785-bucket/
# and trigger a cluster software update.

set -ex

# Install and start SSM agent
if ! systemctl is-active --quiet amazon-ssm-agent 2>/dev/null; then
    echo "Installing SSM agent..."
    snap install amazon-ssm-agent --classic 2>/dev/null \
        || yum install -y amazon-ssm-agent 2>/dev/null \
        || (curl -o /tmp/amazon-ssm-agent.deb https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/debian_amd64/amazon-ssm-agent.deb \
            && dpkg -i /tmp/amazon-ssm-agent.deb)
    systemctl enable amazon-ssm-agent
    systemctl start amazon-ssm-agent
    echo "SSM agent installed and started"
else
    echo "SSM agent already running"
fi
