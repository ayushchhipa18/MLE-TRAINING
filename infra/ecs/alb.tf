resource "aws_lb" "this" {
  name               = "diabetes-alb"
  load_balancer_type = "application"
  subnets            = var.public_subnets
  security_groups    = [var.security_group_id]

  tags = {
    Name = "diabetes-alb"
  }
}
# FastAPI Target Group
resource "aws_lb_target_group" "fastapi" {

  name        = "tg-fastapi"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path     = "/health"
    matcher  = "200"
    interval = 30
    timeout  = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}
# Streamlit Target Group
resource "aws_lb_target_group" "streamlit" {
  name        = "tg-streamlit"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path     = "/"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}
# ALB Listener
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }
}
# /api → FastAPI
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.fastapi.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}
#Listener Rule – Streamlit
resource "aws_lb_listener_rule" "streamlit" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 20

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }
  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}