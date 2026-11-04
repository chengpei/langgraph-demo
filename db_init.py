"""初始化成绩数据库：建表 + 填充示例数据。

用法： .venv/bin/python db_init.py
可重复运行（会先删除旧表重建）。如需保留已有数据，请勿运行。
"""
from db import _conn


DDL = """
CREATE TABLE IF NOT EXISTS teachers (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(50)  NOT NULL COMMENT '姓名',
    gender          VARCHAR(10)  COMMENT '性别',
    age             INT          COMMENT '年龄',
    subject         VARCHAR(50)  COMMENT '科目',
    class_name      VARCHAR(50)  COMMENT '带课班级，如 三年级1班',
    is_head_teacher TINYINT(1)   DEFAULT 0 COMMENT '是否班主任，1是0否',
    phone           VARCHAR(20)  COMMENT '手机号'
) COMMENT='老师表';

CREATE TABLE IF NOT EXISTS students (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(50)  NOT NULL COMMENT '姓名',
    gender          VARCHAR(10)  COMMENT '性别',
    class_name      VARCHAR(50)  COMMENT '班级，如 三年级1班',
    head_teacher_id INT          NULL COMMENT '班主任，对应 teachers.id',
    INDEX idx_class (class_name),
    CONSTRAINT fk_students_head FOREIGN KEY (head_teacher_id) REFERENCES teachers(id)
) COMMENT='学生表';

CREATE TABLE IF NOT EXISTS scores (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT          NOT NULL COMMENT '学生id，对应 students.id',
    exam_month INT          NOT NULL COMMENT '考试月份 1-12',
    subject    VARCHAR(50)  NOT NULL COMMENT '科目',
    score      DECIMAL(5,1) NOT NULL DEFAULT 0 COMMENT '分数',
    INDEX idx_student (student_id),
    INDEX idx_month (exam_month),
    CONSTRAINT fk_scores_student FOREIGN KEY (student_id) REFERENCES students(id)
) COMMENT='考试成绩表';
"""


def main():
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            # deps 顺序：先删依赖多的表
            for t in ("scores", "students", "teachers"):
                cur.execute(f"DROP TABLE IF EXISTS {t}")
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            for stmt in DDL.split(";"):
                if stmt.strip():
                    cur.execute(stmt)
            _seed(cur)
        conn.commit()
        print("建表并填充示例数据完成。")
        _verify(conn)
    finally:
        conn.close()


def _seed(cur):
    # teachers: (name, gender, age, subject, class_name, is_head_teacher, phone)
    teachers = [
        ("王丹", "女", 38, "数学", "三年级1班", 1, "13800001101"),
        ("李强", "男", 35, "语文", "三年级1班", 0, "13800001102"),
        ("陈静", "女", 33, "英语", "三年级2班", 1, "13800001103"),
        ("赵磊", "男", 40, "数学", "三年级2班", 0, "13800001104"),
        ("刘芳", "女", 36, "语文", "四年级1班", 1, "13800001105"),
        ("孙浩", "男", 42, "数学", "四年级1班", 0, "13800001106"),
        ("周悦", "女", 31, "英语", "四年级1班", 0, "13800001107"),
        ("吴敏", "女", 34, "数学", "四年级2班", 1, "13800001108"),
        ("郑强", "男", 37, "语文", "四年级2班", 0, "13800001109"),
    ]
    for name, gender, age, subject, class_name, is_ht, phone in teachers:
        cur.execute(
            "INSERT INTO teachers (name, gender, age, subject, class_name, is_head_teacher, phone)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (name, gender, age, subject, class_name, is_ht, phone),
        )

    # students: (name, gender, class_name) -> head_teacher: 各班的班主任 id
    head_ids = {  # class_name -> teacher id
        "三年级1班": 1, "三年级2班": 3, "四年级1班": 5, "四年级2班": 8,
    }
    students = [
        ("三年级1班", [("张三", "男"), ("李四", "男"), ("王五", "女"), ("赵六", "女")]),
        ("三年级2班", [("钱七", "男"), ("孙八", "女"), ("周九", "男")]),
        ("四年级1班", [("吴十", "男"), ("郑十一", "女"), ("王十二", "男")]),
        ("四年级2班", [("冯十三", "女"), ("陈十四", "男"), ("褚十五", "女")]),
    ]
    student_ids = {}
    for cls, names in students:
        for name, gender in names:
            cur.execute(
                "INSERT INTO students (name, gender, class_name, head_teacher_id)"
                " VALUES (%s,%s,%s,%s)",
                (name, gender, cls, head_ids[cls]),
            )
            student_ids[name] = cur.lastrowid

    # scores：为每个学生生成 7~10 月、语文/数学/英语 的成绩
    # 基础能力分，让例子的"最高/第一名"有清晰答案
    skill = {
        "张三": 85, "李四": 79, "王五": 90, "赵六": 76,
        "钱七": 82, "孙八": 88, "周九": 70,
        "吴十": 84, "郑十一": 91, "王十二": 77,
        "冯十三": 86, "陈十四": 80, "褚十五": 93,
    }
    # (month, subject) -> 相对基础的波动
    delta = {
        (7, "语文"): 6, (7, "数学"): 2, (7, "英语"): 4,
        (8, "语文"): -3, (8, "数学"): 5, (8, "英语"): 1,
        (9, "语文"): 4, (9, "数学"): -2, (9, "英语"): 6,
        (10, "语文"): 2, (10, "数学"): 6, (10, "英语"): -4,
    }
    subjects = ["语文", "数学", "英语"]
    for name, sid in student_ids.items():
        for month in (7, 8, 9, 10):
            for subj in subjects:
                v = skill[name] + delta[(month, subj)]
                v = max(60, min(100, v))
                cur.execute(
                    "INSERT INTO scores (student_id, exam_month, subject, score)"
                    " VALUES (%s,%s,%s,%s)",
                    (sid, month, subj, float(v)),
                )


def _verify(conn):
    with conn.cursor() as cur:
        for t in ("teachers", "students", "scores"):
            cur.execute(f"SELECT COUNT(*) AS n FROM {t}")
            print(f"  {t}: {cur.fetchone()['n']} 行")
        print("---- 示例题目自检 ----")
        checks = [
            ("张三8月总分", "SELECT SUM(score) AS total FROM scores s JOIN students st ON s.student_id=st.id WHERE st.name='张三' AND s.exam_month=8"),
            ("三年级1班9月总分最高", "SELECT st.name, ROUND(SUM(s.score),1) total FROM scores s JOIN students st ON s.student_id=st.id WHERE st.class_name='三年级1班' AND s.exam_month=9 GROUP BY st.id HAVING SUM(s.score) IS NOT NULL ORDER BY total DESC LIMIT 1"),
            ("四年级10月总分第一", "SELECT st.name, ROUND(SUM(s.score),1) total FROM scores s JOIN students st ON s.student_id=st.id WHERE st.class_name LIKE '四年级%' AND s.exam_month=10 GROUP BY st.id ORDER BY total DESC LIMIT 1"),
        ]
        for label, sql in checks:
            cur.execute(sql)
            print(f"  {label}: {cur.fetchall()}")


if __name__ == "__main__":
    main()