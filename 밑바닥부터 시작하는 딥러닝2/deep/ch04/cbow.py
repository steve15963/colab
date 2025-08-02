# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    """
    CBOW (Continuous Bag of Words) 클래스
    주변 단어들로부터 중앙 단어를 예측하는 신경망 모델
    
    예시: "철수는 지금 학교에 갑니다"에서
    - window_size=1일 때: ["철수는", "학교에"] → "지금" 예측
    - window_size=2일 때: ["철수는", "지금", "갑니다", "학교에"] → "학교에" 예측
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        """
        CBOW 모델 초기화 (생성자)
        
        Args:
            vocab_size: 전체 어휘 개수 (예: 10000개 단어)
            hidden_size: 숨겨진 층의 크기 (단어 벡터의 차원, 예: 100차원)
            window_size: 문맥 창 크기 (앞뒤로 몇 개 단어를 볼지, 예: 2)
            corpus: 훈련할 텍스트 데이터
        """
        V, H = vocab_size, hidden_size  # V=어휘개수, H=숨겨진층크기 (편의를 위해 별칭 사용)

        # 가중치 초기화 (학습할 매개변수들)
        # W_in: 입력 가중치 행렬 (단어 → 벡터로 변환하는 표)
        # W_out: 출력 가중치 행렬 (벡터 → 단어 확률로 변환하는 표)
        # 0.01 * np.random.randn(): 작은 랜덤 값으로 초기화 (학습 시작점)
        W_in = 0.01 * np.random.randn(V, H).astype('f')   # shape: (10000, 100) - 각 단어를 100차원 벡터로
        W_out = 0.01 * np.random.randn(V, H).astype('f')  # shape: (10000, 100) - 100차원 벡터를 단어 확률로

        # 입력 레이어들 생성 (문맥 단어들을 처리할 레이어들)
        # window_size=2일 때: 앞 2개 + 뒤 2개 = 총 4개의 문맥 단어
        # 각 문맥 단어마다 별도의 Embedding 레이어가 필요
        self.in_layers = []  # 빈 리스트로 시작
        for i in range(2 * window_size):  # window_size=2면 range(4) → 0,1,2,3
            layer = Embedding(W_in)  # 각 문맥 단어를 벡터로 변환하는 레이어 생성
            self.in_layers.append(layer)  # 리스트에 추가
        
        # 네거티브 샘플링 손실 레이어 생성 (학습 효율성을 위한 기법)
        # power=0.75: 빈도가 높은 단어의 선택 확률을 낮춤 (균형 맞추기)
        # sample_size=5: 네거티브 샘플 5개 사용 (전체 어휘 대신 일부만 비교)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 하나의 리스트로 모음 (학습을 위해)
        # params: 학습할 가중치들, grads: 가중치 업데이트에 필요한 기울기들
        layers = self.in_layers + [self.ns_loss]  # 입력 레이어들 + 손실 레이어
        self.params, self.grads = [], []  # 빈 리스트로 시작
        for layer in layers:  # 각 레이어에서 가중치와 기울기 가져오기
            self.params += layer.params  # 학습할 가중치들 추가
            self.grads += layer.grads    # 기울기들 추가

        # 단어 벡터를 멤버 변수로 설정 (나중에 단어 벡터를 가져올 때 사용)
        # W_in이 바로 단어들의 벡터 표현 (각 행이 하나의 단어 벡터)
        self.word_vecs = W_in

    def forward(self, contexts, target):
        """
        순전파: 문맥 단어들로부터 중앙 단어 예측
        
        Args:
            contexts: 문맥 단어들의 인덱스 (예: [[1, 3, 5, 7]]) - shape: (배치크기, 문맥단어수)
            target: 예측할 중앙 단어의 인덱스 (예: [4]) - shape: (배치크기,)
        
        Returns:
            loss: 예측 오차 (손실값)
        """
        h = 0  # 문맥 벡터들의 합을 저장할 변수 (0으로 초기화)
        
        # 각 문맥 단어를 해당하는 Embedding 레이어로 처리
        for i, layer in enumerate(self.in_layers):  # i=0,1,2,3 (window_size=2일 때)
            # contexts[:, i]: i번째 문맥 단어의 인덱스
            # layer.forward(): 단어 인덱스를 벡터로 변환
            h += layer.forward(contexts[:, i])  # 벡터들을 하나씩 더함
        
        # 문맥 벡터들의 평균 계산 (더 안정적인 학습을 위해)
        h *= 1 / len(self.in_layers)  # h = h / 4 (window_size=2일 때)
        
        # 네거티브 샘플링 손실 계산 (예측 오차 측정)
        # h: 문맥 벡터들의 평균, target: 정답 단어
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        """
        역전파: 오차를 뒤로 전달하면서 가중치 업데이트
        
        Args:
            dout: 뒤에서 넘어온 기울기 (기본값=1, 손실 함수에서 시작)
        
        Returns:
            None (더 이상 뒤로 전달할 기울기가 없음)
        """
        # 네거티브 샘플링 레이어에서 기울기 계산
        dout = self.ns_loss.backward(dout)  # 손실 레이어에서 기울기 받기
        
        # 문맥 벡터 평균에 대한 기울기 계산
        dout *= 1 / len(self.in_layers)  # 평균을 취했으므로 기울기도 나누기
        
        # 각 입력 레이어로 기울기 전달 (가중치 업데이트)
        for layer in self.in_layers:  # 각 Embedding 레이어에 기울기 전달
            layer.backward(dout)  # 가중치 업데이트
        
        return None  # 더 이상 뒤로 전달할 기울기가 없음
