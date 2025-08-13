from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='C:/Users/duanm/Music/GitHubProjects/MLBase/.env') 

print(os.getenv("email"))
print(os.getenv("port"))
print(os.getenv("host"))