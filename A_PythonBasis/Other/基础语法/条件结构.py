
grade = 44.5

if grade > 90:
    print("优秀")
else:
    if grade > 80:
        print("良好")
    else:
        if grade > 60:
            print('及格')
        else:
            print('不及格')

# 与上等价(但更加简洁)
if grade > 90:
    print("优秀")
elif grade > 80:
    print('良好')
elif grade > 60:
    print('及格')
else:
    print('不及格')