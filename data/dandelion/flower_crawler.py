import requests
from parsel import Selector

if __name__=="__main__":

    urls = [f"https://sc.chinaz.com/tupian/pugongyingtupian_{i}.html" for i in range(2,8)]

    header={
        "user-agent":"*"
    }

    n = 1

    for url in urls:
        response_all = requests.get(url,headers=header)
        response_all.encoding = "utf-8"
        # print(response_all.text)

        selector = Selector(response_all.text)
        imgs = selector.xpath("/html/body/div[3]/div[2]//@data-original").getall()
        # print(imgs)

        for img in imgs:
            response = requests.get("https:"+img)
            with open(f"./img{n}.jpg","wb") as file:
                file.write(response.content)
            n += 1