resource "aws_ecs_task_definition" "diabetes_task" {
  family                   = "diabetes-app-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"

  execution_role_arn = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "uvicorn"
      image     = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/mle-uvicorn:latest"
      essential = true

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        },
      ]
    },
    {
      name      = "streamlit"
      image     = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/mle-streamlit:latest"
      essential = true
      
      portMappings = [
        {
          containerPort = 8501
          hostPort      = 8501
          protocol      = "tcp"
        }
      ]
    }
  ])
}

 