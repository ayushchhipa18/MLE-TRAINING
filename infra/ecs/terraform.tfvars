aws_region = "ap-south-1"

vpc_id = "vpc-01e4356a238198e4c"

# ALB (public internet facing)
public_subnets = [
  "subnet-0c146b90ec457e027",
  "subnet-0401edbfdab3db78b"
]

# ECS Fargate
private_subnets = [
  "subnet-0163af6ef9f94c8df"
]

alb_sg = "sg-0eaaf28ccdeb85470"
ecs_sg = "sg-03b352971c446ad20"

ecr_image_uri = "616426097428.dkr.ecr.ap-south-1.amazonaws.com/diabetes-app:latest"
