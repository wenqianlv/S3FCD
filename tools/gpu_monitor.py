import os

import subprocess
import time


import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
 
# 第三方 SMTP 服务
mail_host = "smtp.qq.com"      # SMTP服务器
mail_user = "1178396201@qq.com"                  # 用户名
mail_pass = "mgkjoyidyadfhhhg"               # 授权密码，非登录密码
 
sender = "1178396201@qq.com"    # 发件人邮箱(最好写全, 不然会失败)
receiver = "1178396201@qq.com"    # 发件人邮箱(最好写全, 不然会失败)

FILE_PATH = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(FILE_PATH)
CMD = f"cd  {CUR_DIR}/../ && ls ./ -lrth &&\
    bash tools/train_multi.sh > work_dirs/logs/train_20230417.log 2>&1 &"
 
def sendEmail(title, content):
 
    message = MIMEText(content, 'plain', 'utf-8')  # 内容, 格式, 编码
    message['From'] = "{}".format(sender)
    message['To'] = receiver
    message['Subject'] = title
 
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 启用SSL发信, 端口一般是465
        smtpObj.login(mail_user, mail_pass)  # 登录验证
        smtpObj.sendmail(sender, [receiver], message.as_string())  # 发送
        print("mail has been send successfully.")
    except smtplib.SMTPException as e:
        print(e)


def get_gpu_utilization():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8')
        # print(result)
        memory_data = [int(y) for x in result.strip().split('\n') for y in x.split(',')]
        used_memory = memory_data[::2]
        total_memory = memory_data[1::2]
        gpu_utilization = [used / total * 100 for used, total in zip(used_memory, total_memory)]
        return gpu_utilization
    except subprocess.CalledProcessError as e:
        print(e)
        return None


# def get_gpu_utilization():
#     result = subprocess.run(
#         ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
#         stdout=subprocess.PIPE,
#         universal_newlines=True
#     )
#     utilization = [int(x.strip().rstrip('%')) for x in result.stdout.split('\n') if x.strip()]
#     return utilization

def main():
    # 检查10分钟内GPU利用率是否小于50%
    alert_time = 10 * 60  # 10分钟，单位为秒
    low_utilization_threshold = 50  # 50%
    send_flag = False
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > alert_time:
            # 达到预警时间
            utilization = get_gpu_utilization()
            if any(u < low_utilization_threshold for u in utilization):
                # GPU利用率持续低于50%
                
                if not send_flag:
                    os.system(CMD)
                    start_time = time.time()
                    send_flag = True
                    continue
                print(f'GPU内存利用率低于{low_utilization_threshold}%，进行预警！')
                sendEmail("GPU利用率过低，进行预警！",\
                    f"GPU利用率过低，进行预警！当前利用率：{utilization}。")
                break
            else:
                # GPU利用率正常
                start_time = time.time()
        else:
            # 继续检查GPU利用率
            utilization = get_gpu_utilization()
            print(f'当前GPU利用率: {utilization}')
            
            time.sleep(60)  # 每分钟检查一次

if __name__ == '__main__':
    main()
