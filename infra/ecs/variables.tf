# ---------------------------
# AWS
# ---------------------------
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}
# ---------------------------
# Networking
# ---------------------------
variable "vpc_id" {
  description = "VPC ID Where ECS & ALB will be deployed"
  type        = string
}
variable "public_subnets" {
  description = "Public subnets for ALB"
  type        = list(string)
}
variable "private_subnets" {
  description = "Private subnets for ECS tasks"
  type        = list(string)
}
# ---------------------------
# Security Groups
# ---------------------------
variable "alb_sg" {
  description = "Security group ID for Application Load Balancer"
  type        = string
}

variable "ecs_sg" {
  description = "Security group ID for ECS service"
  type        = string
}

# ---------------------------
# ECS / ECR
# ---------------------------

variable "ecr_image_uri" {
  description = "Full ECR image URI (repo:tag)"
  type        = string
}

variable "ecs_cluster_name" {
  description = "ECS CLuster name"
  type        = string
  default     = "diabetes-ecs-cluster"
}

variable "ecs_service_name" {
  description = "ECS Service name"
  type        = string
  default     = "diabetes-ecs-service"
}

variable "ecs_task_family" {
  description = "ECS Task Definition family name"
  type        = string
  default     = "diabetes-task"
}

# ---------------------------
# ECS Sizing
# ---------------------------
variable "ecs_cpu" {
  description = "CPU units for Fargate task"
  type        = string
  default     = "512"

}

variable "ecs_memory" {
  description = "Memory (MB) for Fargate task"
  type        = string
  default     = "1024"
}
# ---------------------------
# App Ports
# ---------------------------
variable "fastapi_port" {
  description = "FastAPI container port"
  type        = number
  default     = 8000
}

variable "streamlit_port" {
  description = "Streamlit container port"
  type        = number
  default     = 8501
}