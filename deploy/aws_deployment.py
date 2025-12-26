"""
AWS Deployment Configuration and Scripts
Handles deployment to EC2, Lambda, S3, and SageMaker
"""

import os
import json
import boto3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

EC2_USER_DATA = '''#!/bin/bash
set -e

# Update system
yum update -y
yum install -y python3.9 python3.9-pip git

# Install system dependencies for TA-Lib
yum install -y gcc gcc-c++ make wget
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# Create app directory
mkdir -p /opt/trading
cd /opt/trading

# Clone or copy application
# git clone <your-repo> .

# Install Python dependencies
pip3.9 install -r requirements.txt

# Setup environment variables
cat > /opt/trading/.env << 'EOF'
MT5_LOGIN=${MT5_LOGIN}
MT5_PASSWORD=${MT5_PASSWORD}
MT5_SERVER=${MT5_SERVER}
AWS_REGION=${AWS_REGION}
EOF

# Create systemd service
cat > /etc/systemd/system/trading.service << 'EOF'
[Unit]
Description=Aladdin Trading System
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/trading
ExecStart=/usr/bin/python3.9 main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable trading
systemctl start trading
'''

LAMBDA_HANDLER = '''
import json
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda handler for scheduled tasks:
    - Daily model retraining trigger
    - News sentiment fetching
    - Health check alerts
    """
    task = event.get('task', 'health_check')
    
    if task == 'retrain':
        return trigger_retraining()
    elif task == 'fetch_news':
        return fetch_news_sentiment()
    elif task == 'health_check':
        return check_system_health()
    else:
        return {'statusCode': 400, 'body': f'Unknown task: {task}'}

def trigger_retraining():
    """Trigger model retraining on SageMaker"""
    sagemaker = boto3.client('sagemaker')
    
    training_job_name = f"forex-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # This would trigger a SageMaker training job
    # Simplified for demonstration
    
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Retraining triggered', 'job': training_job_name})
    }

def fetch_news_sentiment():
    """Fetch and process news for sentiment analysis"""
    import requests
    
    # Fetch from RSS feeds
    feeds = [
        'https://feeds.bbci.co.uk/news/business/rss.xml',
        'https://www.reuters.com/arc/outboundfeeds/news-feed/'
    ]
    
    articles = []
    for feed_url in feeds:
        try:
            # Would parse RSS feed here
            pass
        except Exception as e:
            print(f"Error fetching {feed_url}: {e}")
    
    # Store in S3
    s3 = boto3.client('s3')
    bucket = os.environ.get('DATA_BUCKET', 'forex-trading-data')
    
    s3.put_object(
        Bucket=bucket,
        Key=f"news/{datetime.now().strftime('%Y/%m/%d')}/articles.json",
        Body=json.dumps(articles)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'News fetched', 'count': len(articles)})
    }

def check_system_health():
    """Check trading system health"""
    ec2 = boto3.client('ec2')
    cloudwatch = boto3.client('cloudwatch')
    sns = boto3.client('sns')
    
    # Check EC2 instance status
    # This would check the trading server status
    
    return {
        'statusCode': 200,
        'body': json.dumps({'status': 'healthy'})
    }
'''

CLOUDFORMATION_TEMPLATE = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "Aladdin Forex Trading System Infrastructure",
    "Parameters": {
        "InstanceType": {
            "Type": "String",
            "Default": "t3.micro",
            "AllowedValues": ["t3.micro", "t3.small", "t3.medium"],
            "Description": "EC2 instance type"
        },
        "KeyName": {
            "Type": "AWS::EC2::KeyPair::KeyName",
            "Description": "SSH key pair name"
        },
        "MT5Login": {
            "Type": "String",
            "NoEcho": True,
            "Description": "MT5 account login"
        },
        "MT5Password": {
            "Type": "String",
            "NoEcho": True,
            "Description": "MT5 account password"
        }
    },
    "Resources": {
        "TradingVPC": {
            "Type": "AWS::EC2::VPC",
            "Properties": {
                "CidrBlock": "10.0.0.0/16",
                "EnableDnsHostnames": True,
                "Tags": [{"Key": "Name", "Value": "TradingVPC"}]
            }
        },
        "PublicSubnet": {
            "Type": "AWS::EC2::Subnet",
            "Properties": {
                "VpcId": {"Ref": "TradingVPC"},
                "CidrBlock": "10.0.1.0/24",
                "MapPublicIpOnLaunch": True
            }
        },
        "InternetGateway": {
            "Type": "AWS::EC2::InternetGateway"
        },
        "AttachGateway": {
            "Type": "AWS::EC2::VPCGatewayAttachment",
            "Properties": {
                "VpcId": {"Ref": "TradingVPC"},
                "InternetGatewayId": {"Ref": "InternetGateway"}
            }
        },
        "TradingSecurityGroup": {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": "Trading system security group",
                "VpcId": {"Ref": "TradingVPC"},
                "SecurityGroupIngress": [
                    {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "CidrIp": "0.0.0.0/0"},
                    {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "CidrIp": "0.0.0.0/0"}
                ]
            }
        },
        "TradingInstance": {
            "Type": "AWS::EC2::Instance",
            "Properties": {
                "InstanceType": {"Ref": "InstanceType"},
                "KeyName": {"Ref": "KeyName"},
                "ImageId": "ami-0c55b159cbfafe1f0",
                "SubnetId": {"Ref": "PublicSubnet"},
                "SecurityGroupIds": [{"Ref": "TradingSecurityGroup"}],
                "IamInstanceProfile": {"Ref": "TradingInstanceProfile"},
                "Tags": [{"Key": "Name", "Value": "TradingServer"}]
            }
        },
        "TradingRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                },
                "ManagedPolicyArns": [
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
                ]
            }
        },
        "TradingInstanceProfile": {
            "Type": "AWS::IAM::InstanceProfile",
            "Properties": {
                "Roles": [{"Ref": "TradingRole"}]
            }
        },
        "DataBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": {"Fn::Sub": "forex-trading-data-${AWS::AccountId}"},
                "VersioningConfiguration": {"Status": "Enabled"}
            }
        },
        "ModelBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": {"Fn::Sub": "forex-trading-models-${AWS::AccountId}"}
            }
        },
        "DailyRetrainRule": {
            "Type": "AWS::Events::Rule",
            "Properties": {
                "Description": "Trigger daily model retraining",
                "ScheduleExpression": "cron(0 0 * * ? *)",
                "State": "ENABLED",
                "Targets": [{
                    "Id": "RetrainTarget",
                    "Arn": {"Fn::GetAtt": ["RetrainLambda", "Arn"]},
                    "Input": "{\"task\": \"retrain\"}"
                }]
            }
        },
        "RetrainLambda": {
            "Type": "AWS::Lambda::Function",
            "Properties": {
                "FunctionName": "forex-retrain-trigger",
                "Runtime": "python3.9",
                "Handler": "index.lambda_handler",
                "Role": {"Fn::GetAtt": ["LambdaRole", "Arn"]},
                "Timeout": 300,
                "Code": {
                    "ZipFile": LAMBDA_HANDLER
                }
            }
        },
        "LambdaRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                },
                "ManagedPolicyArns": [
                    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                ]
            }
        },
        "TradingAlarm": {
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": "TradingServerDown",
                "MetricName": "StatusCheckFailed",
                "Namespace": "AWS/EC2",
                "Statistic": "Maximum",
                "Period": 300,
                "EvaluationPeriods": 2,
                "Threshold": 1,
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
                "Dimensions": [{
                    "Name": "InstanceId",
                    "Value": {"Ref": "TradingInstance"}
                }]
            }
        }
    },
    "Outputs": {
        "InstanceId": {
            "Value": {"Ref": "TradingInstance"},
            "Description": "Trading server instance ID"
        },
        "PublicIP": {
            "Value": {"Fn::GetAtt": ["TradingInstance", "PublicIp"]},
            "Description": "Trading server public IP"
        },
        "DataBucketName": {
            "Value": {"Ref": "DataBucket"},
            "Description": "S3 bucket for trading data"
        },
        "ModelBucketName": {
            "Value": {"Ref": "ModelBucket"},
            "Description": "S3 bucket for ML models"
        }
    }
}


class AWSDeployer:
    """AWS deployment manager"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.session = boto3.Session(region_name=region)
    
    def deploy_infrastructure(self, stack_name: str, params: dict):
        """Deploy CloudFormation stack"""
        cf = self.session.client('cloudformation')
        
        template_body = json.dumps(CLOUDFORMATION_TEMPLATE)
        
        parameters = [
            {'ParameterKey': k, 'ParameterValue': v}
            for k, v in params.items()
        ]
        
        try:
            cf.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_IAM']
            )
            logger.info(f"Stack {stack_name} creation initiated")
            
            # Wait for completion
            waiter = cf.get_waiter('stack_create_complete')
            waiter.wait(StackName=stack_name)
            
            logger.info(f"Stack {stack_name} created successfully")
            
            # Get outputs
            response = cf.describe_stacks(StackName=stack_name)
            outputs = {
                o['OutputKey']: o['OutputValue']
                for o in response['Stacks'][0]['Outputs']
            }
            
            return outputs
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def upload_code_to_s3(self, bucket: str, local_path: str, s3_prefix: str = 'code'):
        """Upload application code to S3"""
        s3 = self.session.client('s3')
        
        import zipfile
        import io
        
        # Create zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file.endswith('.py') or file == 'requirements.txt':
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, local_path)
                        zf.write(file_path, arcname)
        
        zip_buffer.seek(0)
        
        # Upload to S3
        key = f"{s3_prefix}/trading-system-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        s3.upload_fileobj(zip_buffer, bucket, key)
        
        logger.info(f"Code uploaded to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"
    
    def create_sagemaker_training_job(self, job_name: str, s3_data_path: str,
                                       s3_output_path: str, instance_type: str = 'ml.m5.large'):
        """Create SageMaker training job"""
        sm = self.session.client('sagemaker')
        
        training_params = {
            'TrainingJobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-cpu-py38',
                'TrainingInputMode': 'File'
            },
            'RoleArn': 'arn:aws:iam::ACCOUNT_ID:role/SageMakerRole',
            'InputDataConfig': [{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_data_path
                    }
                }
            }],
            'OutputDataConfig': {
                'S3OutputPath': s3_output_path
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': 50
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            }
        }
        
        try:
            sm.create_training_job(**training_params)
            logger.info(f"Training job {job_name} created")
            return job_name
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise
    
    def setup_cloudwatch_dashboard(self, dashboard_name: str, instance_id: str):
        """Create CloudWatch dashboard for monitoring"""
        cw = self.session.client('cloudwatch')
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "CPU Utilization",
                        "metrics": [
                            ["AWS/EC2", "CPUUtilization", "InstanceId", instance_id]
                        ],
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "Network Traffic",
                        "metrics": [
                            ["AWS/EC2", "NetworkIn", "InstanceId", instance_id],
                            ["AWS/EC2", "NetworkOut", "InstanceId", instance_id]
                        ],
                        "period": 300
                    }
                },
                {
                    "type": "log",
                    "x": 0, "y": 6, "width": 24, "height": 6,
                    "properties": {
                        "title": "Trading Logs",
                        "query": "SOURCE '/var/log/trading.log' | fields @timestamp, @message | sort @timestamp desc | limit 100"
                    }
                }
            ]
        }
        
        cw.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        
        logger.info(f"Dashboard {dashboard_name} created")


def print_deployment_guide():
    """Print deployment guide"""
    guide = """
================================================================================
ALADDIN FOREX TRADING SYSTEM - AWS DEPLOYMENT GUIDE
================================================================================

PREREQUISITES:
1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. SSH key pair created in AWS
4. Python 3.9+ installed locally

QUICK START:
-------------

1. Configure AWS credentials:
   $ aws configure
   
2. Create the infrastructure:
   $ python deploy/aws_deployment.py --deploy --stack-name forex-trading
   
3. Upload application code:
   $ python deploy/aws_deployment.py --upload --bucket <your-bucket>
   
4. SSH into the trading server:
   $ ssh -i your-key.pem ec2-user@<public-ip>
   
5. Start the trading system:
   $ sudo systemctl start trading
   $ sudo systemctl status trading

MANUAL DEPLOYMENT STEPS:
------------------------

1. Create EC2 Instance:
   - AMI: Amazon Linux 2
   - Instance Type: t3.micro (free tier) or t3.small
   - Security Group: Allow SSH (22), HTTPS (443)
   - IAM Role: With S3 and CloudWatch access

2. Install Dependencies:
   $ sudo yum update -y
   $ sudo yum install -y python3.9 python3.9-pip git
   $ pip3.9 install -r requirements.txt

3. Configure Environment:
   $ cat > .env << EOF
   MT5_LOGIN=your_login
   MT5_PASSWORD=your_password
   MT5_SERVER=MetaQuotes-Demo
   EOF

4. Setup Systemd Service:
   $ sudo cp trading.service /etc/systemd/system/
   $ sudo systemctl daemon-reload
   $ sudo systemctl enable trading
   $ sudo systemctl start trading

5. Setup CloudWatch Logs:
   - Install CloudWatch agent
   - Configure log streaming

COST ESTIMATION (Monthly):
--------------------------
- EC2 t3.micro: ~$8.50 (or free tier)
- S3 storage: ~$0.50 (minimal data)
- CloudWatch: ~$1.00
- Lambda: Free tier
- Total: ~$10/month

MONITORING:
-----------
- CloudWatch Dashboard: AWS Console > CloudWatch > Dashboards
- Logs: /var/log/trading.log
- Alerts: Configure SNS for email/SMS notifications

SECURITY BEST PRACTICES:
------------------------
1. Never commit credentials to git
2. Use AWS Secrets Manager for sensitive data
3. Enable VPC flow logs
4. Use IAM roles instead of access keys
5. Enable CloudTrail for audit logging
6. Regularly rotate credentials

TROUBLESHOOTING:
----------------
1. Check service status:
   $ sudo systemctl status trading
   
2. View logs:
   $ sudo journalctl -u trading -f
   $ tail -f /opt/trading/logs/trading.log
   
3. Restart service:
   $ sudo systemctl restart trading

4. Check connectivity:
   $ curl -I https://api.mt5.com

================================================================================
"""
    print(guide)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Deployment for Trading System')
    parser.add_argument('--deploy', action='store_true', help='Deploy infrastructure')
    parser.add_argument('--upload', action='store_true', help='Upload code to S3')
    parser.add_argument('--guide', action='store_true', help='Print deployment guide')
    parser.add_argument('--stack-name', type=str, default='forex-trading')
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    if args.guide:
        print_deployment_guide()
    elif args.deploy:
        deployer = AWSDeployer(region=args.region)
        # Would need actual parameters
        print("Deployment requires AWS credentials and parameters")
        print("Run with --guide for manual deployment instructions")
    elif args.upload:
        if not args.bucket:
            print("--bucket required for upload")
        else:
            deployer = AWSDeployer(region=args.region)
            deployer.upload_code_to_s3(args.bucket, '.', 'code')
    else:
        print_deployment_guide()
