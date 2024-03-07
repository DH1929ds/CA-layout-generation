from diffusers import DDPMScheduler


class MyDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.alphas_cumprod)
        # 부모 클래스의 생성자 호출
        # 여기에 추가적인 초기화 코드를 작성할 수 있습니다.

    # 여기에 추가적인 메소드를 정의할 수 있습니다.

my = MyDDPMScheduler()

