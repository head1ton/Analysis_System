import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        # 크롤링 할 주제의 링크 페이지 주소 css
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            # 크롤링 할 주제의 링크 페이지 주소로 이동하는 페이지 주소(크롤링 할 내용의 페이지 주소)
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href, callback=self.parse_jutsu)
            yield extracted_data

        # 다음 페이지로 이동 class css
        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)

    def parse_jutsu(self, response):
        # 크롤링 내용의 제목 css
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0]
        jutsu_name = jutsu_name.strip()

        # 크롤링 내용본문 css
        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        jutsu_type = ""
        if soup.find('aside'):
            aside = soup.find('aside')

            for cell in aside.find_all('div', {'class':'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip()

        # 'aside' 태그를 제거
        soup.find('aside').decompose()

        # 본문 텍스트를 추출하고 'Trivia' 이전의 내용만 사용
        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict(
            jutus_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )
