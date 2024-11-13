### 快速部署说明

本文件提供了如何根据提供的源代码对项目进行快速部署的指导，包括Web服务和Socket套接字服务。部署的相关源代码可以在 `deploy/deploy_server.py` 和`deploy/deploy_socket.py` 中找到。

#### 如何运作

`deploy_server.py` 和`deploy_socket.py` 分别依据Flask网络服务和Python的套接字包进行编写。

- `deploy_server.py` ：通过Flask进行服务的创建，设置`@app.route`路由进行监听，获取到请求信息则进行深度学习的推理。  

  **此处为详细代码**：

  ```python
  # 主方法
  app.run(host=host, port=port, debug=False)
  
  # 路由
  @app.route("/api/generate/df", methods=["POST"])
  def generate_diffusion_model_api():
      data = request.json
      # 其它代码...
  ```

- `deploy_socket.py`：通过创建套接字并进行端口绑定，监听所绑定端口。若监听到请求，则创建一个新线程进行深度学习的推理。  

  **此处为详细代码**：

  ```python
  # 创建服务套接字
  server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
  # 获取本地host
  host = socket.gethostname()
  # 设置端口
  port = 12345
  # 绑定套接字和本地host与端口映射
  server_socket.bind((host, port))
  # 设置最大监听数量
  server_socket.listen(5)
  # 获取本地服务连接信息
  local_server_address = server_socket.getsockname()
  ```


#### 使用方法

`deploy_server.py` 和`deploy_socket.py` 分别设置`host`和`port`参数即可。注意，端口尽量与常用端口区分，同时两个服务的端口不要设置一样，以防端口占用。

上述方法可以快速的部署你的在线推理服务，也可在**Docker、或其它容器**中快速部署应用。
