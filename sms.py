import pymysql
from langchain.tools import tool

from db import _conn


@tool
def send_sms(phone: str, message: str) -> str:
    """向指定手机号发送一条短信。

    Args:
        phone: 接收短信的手机号（如需发给某位老师，可先查 teachers 表的 phone 字段拿到手机号）
        message: 短信内容
    """
    phone = phone.strip()

    if not phone:
        return "手机号为空，请提供要发送的手机号"

    # 尽力反查该手机号对应哪位老师，便于日志展示；查不到也照常发送
    teacher = _find_teacher_by_phone(phone)
    target = f"{teacher['name']}（{phone}）" if teacher else phone

    # TODO: 此处接入真实短信服务商，当前仅为模拟——打印日志
    print(f"[SMS] 向 {target} 发送短信：{message}")

    return f"已向 {target} 发送短信，内容：{message}"


def _find_teacher_by_phone(phone):
    """按手机号反查老师信息；查不到返回 None。"""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, phone FROM teachers WHERE phone=%s", (phone,))
            return cur.fetchone()
    except pymysql.MySQLError:
        return None
    finally:
        conn.close()