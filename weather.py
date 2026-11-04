from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气情况。

    Args:
        city: 城市名称，例如武汉、北京、上海
    """
    weather_data = {
        "武汉": "晴天，气温 25℃",
        "北京": "多云，气温 20℃",
        "上海": "小雨，气温 22℃",
    }

    return weather_data.get(city, f"没有找到 {city} 的天气数据")