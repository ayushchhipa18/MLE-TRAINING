resource "aws_ecs_service" "this" {
  name            = "diabetes-service"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  enable_execute_command = true

  network_configuration {
  subnets         = var.public_subnets
  security_groups = [var.ecs_sg]
  assign_public_ip = true
}

  # FastAPI → 8000
  load_balancer {
    target_group_arn = aws_lb_target_group.fastapi.arn
    container_name   = "diabetes_container"
    container_port   = 8000
  }

  #Streamlit -> 8501
  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "diabetes_container"
    container_port   = 8501
  }

  depends_on = [
    aws_lb_listener.this,
    aws_lb_listener_rule.api
  ]
}