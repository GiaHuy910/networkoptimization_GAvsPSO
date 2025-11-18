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

#chuyển đổi thành danh sách kề có thể đưa lên excel v_from-v_to-weight
rows=[]
for v_from,edges in G.items():
    for v_to,weight in edges:
        rows.append([v_from,v_to,weight])
df=pd.DataFrame(rows,columns=['v_from','v_to','weight'])

#Đưa lên file Excel
df.to_csv('Graphs_edges.csv',index=False)
#Lời giải là tìm ra đường đi tải được tối đa, return một list path
