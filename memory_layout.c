#include <stdio.h>

int main() {
    int a = 10;
    int b = 20;
    int c = 30;

    int *p1 = &a;
    int *p2 = &b;
    int *p3 = &c;

    printf("변수 주소:\n");
    printf("&a = %p\n", (void*)&a);
    printf("&b = %p\n", (void*)&b);
    printf("&c = %p\n", (void*)&c);
    
    printf("\n포인터 변수 주소:\n");
    printf("&p1 = %p\n", (void*)&p1);
    printf("&p2 = %p\n", (void*)&p2);
    printf("&p3 = %p\n", (void*)&p3);
    
    printf("\n포인터가 가리키는 값:\n");
    printf("p1 = %p (가리키는 값: %d)\n", (void*)p1, *p1);
    printf("p2 = %p (가리키는 값: %d)\n", (void*)p2, *p2);
    printf("p3 = %p (가리키는 값: %d)\n", (void*)p3, *p3);
    
    printf("\n포인터 간 거리:\n");
    printf("&p2 - &p1 = %td\n", &p2 - &p1);
    printf("&p3 - &p2 = %td\n", &p3 - &p2);
    
    printf("\n&p1 + 1의 주소: %p\n", (void*)(&p1 + 1));
    printf("&p2의 주소: %p\n", (void*)&p2);
    printf("&p1 + 1 == &p2? %s\n", (&p1 + 1) == &p2 ? "예" : "아니오");

    return 0;
} 