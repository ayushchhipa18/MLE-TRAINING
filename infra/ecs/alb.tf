resource "aws_lb" "this" {
  name               = "diabetes-alb"
  load_balancer_type = "application"
  subnets            = var.public_subnets
  security_groups    = [var.alb_sg]
}
# FastAPI Target Group
resource "aws_lb_target_group" "fastapi" {

  name        = "diabetes-fastapi-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path     = "/api/health"
    port     = "8000"
    protocol = "HTTP"
  }
}
# Streamlit Target Group
resource "aws_lb_target_group" "streamlit" {
  name        = "diabetes-streamlit-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path     = "/"
    port     = "8501"
    protocol = "HTTP"
  }
}
# ALB Listener
resource "aws_lb_listener" "this" {
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
  listener_arn = aws_lb_listener.this.arn
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