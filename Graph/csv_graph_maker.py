#v_from : [(v_to,cost)]
#bài mẫu để thử làm và giải , sau khi hoàn thành thuật toán sẽ dùng bài khác
#cần xem video giải bài trên vào link #https://www.youtube.com/watch?v=3dOQ3IoVmvY&t=934s
import pandas as pd 

G = {
        's': [('2', 5), ('4', 5)],
        '2': [('3', 6), ('5', 3)],
        '3': [('t', 6)],
        '4': [('3', 3), ('5', 1)],
        '5': [('t', 6)],
        't': []
}
T = {
    's': [('1', 12), ('2', 10), ('3', 8), ('4', 15)],
    '1': [('5', 10), ('6', 8), ('7', 6)],
    '2': [('6', 5), ('7', 12), ('8', 7)],
    '3': [('8', 10), ('9', 5), ('10', 12)],
    '4': [('9', 10), ('10', 6), ('11', 14)],

    '5': [('12', 10), ('13', 8)],
    '6': [('13', 7), ('14', 10)],
    '7': [('14', 9), ('15', 11)],
    '8': [('15', 10), ('16', 7)],
    '9': [('16', 12), ('17', 8)],
    '10': [('17', 11), ('18', 6)],
    '11': [('18', 9), ('19', 13)],

    '12': [('20', 14)],
    '13': [('20', 10), ('21', 6)],
    '14': [('21', 8), ('22', 10)],
    '15': [('22', 12), ('23', 9)],
    '16': [('23', 11)],
    '17': [('23', 7), ('24', 10)],
    '18': [('24', 9)],
    '19': [('24', 11), ('25', 7)],

    '20': [('t', 15)],
    '21': [('t', 12)],
    '22': [('t', 18)],
    '23': [('t', 14)],
    '24': [('t', 10)],
    '25': [('t', 16)],

    't': []
}
#chuyển đổi thành danh sách kề có thể đưa lên excel v_from-v_to-weight
def write_to_file_excel(path,graph):
    rows=[]
    for v_from,edges in graph.items():
        for v_to,weight in edges:
            rows.append([v_from,v_to,weight])
    df=pd.DataFrame(rows,columns=['v_from','v_to','weight'])

    #Đưa lên file Excel
    df.to_csv(path,index=False)
    #Lời giải là tìm ra đường đi tải được tối đa, return một list path

write_to_file_excel('Graph/GraphT_edges.csv',T)