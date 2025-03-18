import pandas as pd
import networkx as nx
from pyvis.network import Network

class CharacterNetworkGenerator():
    def __init__(self):
        pass

    def generate_character_network(self, df):
        '''
        1. 윈도우 설정: windows=10으로 설정하여, 문장 10개 단위로 관계를 분석합니다. 즉, 현재 문장과 이전 9개 문장 내에 등장하는 인물들 간의 관계를 파악합니다.
        2. 인물 관계 추출:
            df['ners'] 열에는 각 문장에서 추출된 인물 이름들이 리스트 형태로 저장되어 있습니다.
            함수는 이 리스트를 순회하며 각 문장에서 등장하는 인물들을 확인합니다.
            현재 문장과 윈도우 내 이전 문장들에 등장하는 인물들을 비교하여, 서로 다른 인물들 간의 관계를 entity_relationship 리스트에 추가합니다.
            entity_relationship에는 인물 이름 쌍이 저장됩니다 (예: ['Naruto', 'Sasuke']).
        3. 관계 데이터프레임 생성:
            entity_relationship 리스트를 이용하여 pandas DataFrame을 생성합니다.
            'source' 열에는 관계의 시작점이 되는 인물, 'target' 열에는 관계의 끝점이 되는 인물을 저장합니다.
            'value' 열에는 두 인물이 함께 등장한 횟수를 저장합니다.
        4. 데이터프레임 정렬:
            'value' 열을 기준으로 내림차순 정렬하여, 관계가 강한 인물 쌍이 상위에 위치하도록 합니다.
        5. 결과 반환:
            생성된 관계 데이터프레임(relationship_df)을 반환합니다.
        '''
        windows = 10  # 관계 분석 윈도우 크기 (문장 개수)
        entity_relationship = []  # 인물 간 관계를 저장할 리스트

        for row in df['ners']:  # df의 'ners' 열 (각 문장의 인물 정보) 순회
            previous_entities_in_window = []  # 윈도우 내 이전 문장의 인물 정보 저장

            for sentence in row:  # 각 문장의 인물 정보 순회
                previous_entities_in_window.append(list(sentence))  # 현재 문장의 인물 정보 추가
                previous_entities_in_window = previous_entities_in_window[-windows:]  # 윈도우 크기 유지

                # 윈도우 내 모든 인물 정보를 1차원 리스트로 변환
                previous_entities_flattened = sum(previous_entities_in_window,[])

                for entity in sentence:  # 현재 문장의 각 인물 순회
                    for entity_in_window in previous_entities_flattened:  # 윈도우 내 모든 인물 순회
                        if entity != entity_in_window:  # 현재 인물과 윈도우 내 다른 인물 비교
                            entity_relationship.append(sorted([entity, entity_in_window]))  # 관계 추가

        # 관계 정보를 이용하여 데이터프레임 생성
        relationship_df = pd.DataFrame({'value': entity_relationship})
        relationship_df['source'] = relationship_df['value'].apply(lambda x: x[0])  # 관계 시작점
        relationship_df['target'] = relationship_df['value'].apply(lambda x: x[1])  # 관계 끝점
        relationship_df = relationship_df.groupby(['source', 'target']).count().reset_index()  # 관계 횟수 계산
        relationship_df = relationship_df.sort_values('value',ascending=False)  # 관계 강도 순 정렬

        return relationship_df  # 관계 데이터프레임 반환

    def draw_network_graph(self, relationship_df):
        relationship_df = relationship_df.sort_values('value', ascending=False)
        relationship_df = relationship_df.head(200)

        G = nx.from_pandas_edgelist(
            relationship_df,
            source='source',
            target='target',
            edge_attr='value',
            create_using=nx.Graph()
        )

        net = Network(notebook=True, width="1000px", height="700px", bgcolor="#222222", font_color="white", cdn_resources="remote")
        node_degree = dict(G.degree)

        nx.set_node_attributes(G, node_degree, 'size')
        net.from_nx(G)

        html = net.generate_html()
        html = html.replace("'", "\"")

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms
        allow-scripts allow-same-origin allow-popups
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        return output_html