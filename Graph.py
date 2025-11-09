#v_from : [(v_to,cost)]
#bài mẫu để thử làm và giải , sau khi hoàn thành thuật toán sẽ dùng bài khác
#cần xem video giải bài trên vào link #https://www.youtube.com/watch?v=3dOQ3IoVmvY&t=934s

G = {
    's': [('2', 5), ('4', 5)],
    '2': [('3', 6), ('5', 3)],
    '3': [('t', 6)],
    '4': [('3', 3), ('5', 1)],
    '5': [('t', 6)],
    't': []
}

#Lời giải là tìm ra đường đi tải được tối đa, return một list path
