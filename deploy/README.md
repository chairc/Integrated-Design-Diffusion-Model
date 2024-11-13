### Quick Deployment README

This document provides guidance on quickly deploying the project based on the provided source code, including both Web and Socket services. The relevant source code for deployment can be found in `deploy/deploy_server.py` and `deploy/deploy_socket.py`.

#### How It Works

`deploy_server.py` and `deploy_socket.py` are written using Flask for web services and Python's socket library for socket services, respectively.

- `deploy_server.py`: Creates a service using Flask, sets up `@app.route` routes to listen for requests, and performs deep learning inference upon receiving request data.

  **Example Code**:

  ```python
  # Main function
  app.run(host=host, port=port, debug=False)
  
  # Route
  @app.route("/api/generate/df", methods=["POST"])
  def generate_diffusion_model_api():
      data = request.json
      # Other code...
  ```

- `deploy_socket.py`: Creates a socket and binds it to a specified port. When a request is detected, a new thread is created to handle deep learning inference.

  **Example Code**:

  ```python
  # Create server socket
  server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
  # Get localhost name
  host = socket.gethostname()
  # Set port
  port = 12345
  # Bind the socket with localhost and port
  server_socket.bind((host, port))
  # Set the maximum number of listeners
  server_socket.listen(5)
  # Get the connection information of the local server
  local_server_address = server_socket.getsockname()
  ```

#### Usage

Simply set the `host` and `port` parameters in both `deploy_server.py` and `deploy_socket.py`. Ensure that the ports used do not conflict with commonly used ports, and that the ports for these two services are different to avoid port conflicts.

These methods allow you to quickly deploy your online inference service and can also be used to rapidly deploy applications in **Docker or other containers**.