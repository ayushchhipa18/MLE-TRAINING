# ---------------------------
# AWS
# ---------------------------
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}
variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
}

# ---------------------------
# Networking
# ---------------------------
variable "vpc_id" {}
variable "public_subnets" {
  type = list(string)
}
variable "private_subnets" {
  type = list(string)
}
variable "security_group_id" {
  type = string
}




