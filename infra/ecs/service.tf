resource "aws_ecs_service" "diabetes_service" {
  name            = "diabetes-app-service"
  cluster         = aws_ecs_cluster.diabetes_cluster.id
  task_definition = aws_ecs_task_definition.diabetes_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  enable_execute_command = true

  network_configuration {
  subnets         = var.public_subnets
  security_groups = [var.security_group_id]
  assign_public_ip = true
}

  # FastAPI → 8000
  load_balancer {
    target_group_arn = aws_lb_target_group.fastapi.arn
    container_name   = "uvicorn"
    container_port   = 8000
  }

  #Streamlit -> 8501
  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "streamlit"
    container_port   = 8501
  }

  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener_rule.api,
    aws_lb_listener_rule.streamlit
  ]
}