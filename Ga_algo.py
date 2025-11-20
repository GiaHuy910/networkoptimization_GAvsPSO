import random
import copy
import numpy as np

# HÀM 1: Khởi tạo quần thể - đường đi để lai ghép,đột biến...
#keys: tên đỉnh
#values: trọng số các đỉnh kề/chi phí các kết nối
def khoi_tao_quan_the(do_thi, diem_bat_dau, diem_ket_thuc, kich_thuoc_quan_the):
    #Khởi tạo quần thể gồm nhiều đường đi ngẫu nhiên từ start đến end
    quan_the = []
   
    for _ in range(kich_thuoc_quan_the): #không quan tâm chỉ số nên dùng _
        duong_di = _tao_duong_di_ngau_nhien(do_thi, diem_bat_dau, diem_ket_thuc)
        if duong_di:  # Chỉ thêm đường đi hợp lệ
            quan_the.append(duong_di)
   
    # Nếu không tạo đủ, thêm các đường đi đơn giản
    while len(quan_the) < kich_thuoc_quan_the:
        duong_don_gian = [diem_bat_dau, diem_ket_thuc]
        if _kiem_tra_duong_di_hop_le(duong_don_gian, do_thi):
            quan_the.append(duong_don_gian)
        else:
            break
           
    return quan_the


def _tao_duong_di_ngau_nhien(do_thi, diem_bat_dau, diem_ket_thuc):
    #hàm phức tạp phụ trợ cần ẩn đi
    #Tạo đường đi ngẫu nhiên
    if diem_bat_dau not in do_thi:
        return None
       
    duong_di = [diem_bat_dau] #danh sách các đỉnh
    diem_hien_tai = diem_bat_dau
    da_tham = set([diem_bat_dau]) #dùng set vì nó nhanh duyệt luôn hằng số không phải tuần tự.
   
    so_buoc_toi_da = len(do_thi) * 2  # Tránh vòng lặp vô hạn
   
    for _ in range(so_buoc_toi_da):
        if diem_hien_tai == diem_ket_thuc:
            break
        # Lấy các điểm kế tiếp có thể đi
        cac_lua_chon = list(do_thi.get(diem_hien_tai, {}).keys())
        #lấy các đỉnh kề với đỉnh hiện tại
        #nếu diem_hien_tai không có trong do_thi thì trả về {} không mắc lỗi
        if not cac_lua_chon: #nếu không có lựa chọn nào để đi tiếp thì dừng lại
            break
           
        # Ưu tiên điểm chưa thăm
        #lọc ra các điểm chưa thăm
        chua_tham = [diem for diem in cac_lua_chon if diem not in da_tham]
        if chua_tham: #nếu còn điểm chưa thăm thì chọn ngẫu nhiên 1 điểm trong số đó
            diem_tiep_theo = random.choice(chua_tham)
        else:
            diem_tiep_theo = random.choice(cac_lua_chon)
           
        duong_di.append(diem_tiep_theo)
        da_tham.add(diem_tiep_theo)
        diem_hien_tai = diem_tiep_theo
   
    # Chỉ trả về đường đi đến được đích
    return duong_di if duong_di[-1] == diem_ket_thuc else None


def _kiem_tra_duong_di_hop_le(duong_di, do_thi):
    #Kiểm tra đường đi có hợp lệ không
    if len(duong_di) < 2: #ít hơn 2 thì không có gì để tìm
        return False
       
    for i in range(len(duong_di) - 1):
        if duong_di[i+1] not in do_thi.get(duong_di[i], {}):
            #trong đó do_thi.get(duong_di[i], {}): trả về danh sách các đỉnh kề với đỉnh i hiện tại
            #tức là nếu đỉnh i+1 không nằm trong danh sách các đỉnh kề với đỉnh i hiện tại
            return False
    return True

# HÀM 2: Tính fitness
def tinh_fitness(quan_the, do_thi):
    #Tính fitness cho toàn bộ quần thể (nghịch đảo tổng chi phí)
    danh_sach_fitness = []
   
    for duong_di in quan_the:
        fitness = _tinh_fitness_duong_di(duong_di, do_thi)
        danh_sach_fitness.append(fitness)
   
    return danh_sach_fitness


def _tinh_fitness_duong_di(duong_di, do_thi):
    #Tính fitness cho một đường đi cụ thể
    if not duong_di or len(duong_di) < 2:
        #nếu đường đi rỗng hoặc không tồn tại hoặc độ dài đường đi có mỗi 1 nút
        return 0.0
   
    tong_chi_phi = 0.0
    for i in range(len(duong_di) - 1):
        diem_hien_tai = duong_di[i]
        diem_ke_tiep = duong_di[i + 1]
       
        # Kiểm tra kết nối hợp lệ
        if diem_ke_tiep not in do_thi.get(diem_hien_tai, {}):
            return 0.0
           
        tong_chi_phi += do_thi[diem_hien_tai][diem_ke_tiep] #cộng kinh phí vô
   
    # Fitness = nghịch đảo chi phí (chi phí càng thấp càng tốt)
    return 1.0 / tong_chi_phi if tong_chi_phi > 0 else 0.0

# HÀM 3: Chọn lọc
def chon_loc(quan_the, danh_sach_fitness, phuong_phap='tournament'):
    #Chọn lọc cá thể cho thế hệ tiếp theo
   
    if phuong_phap == 'tournament': #chọn ngẫu nhiên nhóm cá thể và chọn cá thể tốt nhất là winner
        return _chon_loc_tournament(quan_the, danh_sach_fitness)
    elif phuong_phap == 'roulette':
        """Mỗi cá thể có một 'miếng bánh' trên bánh xe roulette,
        kích thước miếng bánh tỷ lệ với fitness của cá thể đó.
        Cá thể nào có fitness cao hơn thì có miếng bánh lớn hơn
        và dễ được chọn hơn."""
        return _chon_loc_roulette(quan_the, danh_sach_fitness)
    else:
        return _chon_loc_tournament(quan_the, danh_sach_fitness)  # Mặc định


def _chon_loc_tournament(quan_the, danh_sach_fitness, kich_thuoc_tournament=3):
    quan_the_moi = []
   
    for _ in range(len(quan_the)):
        # Chọn ngẫu nhiên k cá thể
        cac_chi_so = random.sample(range(len(quan_the)), kich_thuoc_tournament)
        #tập hợp lấy mẫu và kích thước lấy mẫu cho tournament
        cac_ca_the = [quan_the[i] for i in cac_chi_so]
        #ở đây ta có  mỗi quần thể là tùy chỉ số tournament nên có 2-3 hoặc nhiều hơn chỉ số [1,2,9,0...]
        #tính fitness của các các chỉ số của các đường đi trong quần thể để tìm ra chỉ số tốt nhất
        cac_fitness = [danh_sach_fitness[i] for i in cac_chi_so]
       
        # Chọn cá thể có fitness cao nhất
        chi_so_tot_nhat = cac_chi_so[np.argmax(cac_fitness)]
        quan_the_moi.append(quan_the[chi_so_tot_nhat])
   
    return quan_the_moi


def _chon_loc_roulette(quan_the, danh_sach_fitness):
    # Tránh chia cho 0
    tong_fitness = sum(danh_sach_fitness)
    if tong_fitness == 0:
        return quan_the.copy()
   
    xac_suat = [fitness / tong_fitness for fitness in danh_sach_fitness]
   
    # Chọn lại với xác suất tỷ lệ fitness
    chi_so_duoc_chon = np.random.choice(
        range(len(quan_the)), #danh sách chỉ số có thể chọn
        size=len(quan_the), #số lượng cần chọn
        p=xac_suat #xác suất cho từng chỉ số
    ) #chọn ngẫu nhiên với xác suất tỷ lệ với fitness
   
    return [quan_the[i] for i in chi_so_duoc_chon]

def _sua_duong_di(duong_di, do_thi, diem_bat_dau, diem_ket_thuc):
    #Sửa chữa đường đi để đảm bảo tính hợp lệ
    if not duong_di or duong_di[0] != diem_bat_dau:
        duong_di = [diem_bat_dau]
   
    # Loại bỏ node trùng lặp, giữ thứ tự đầu tiên
    da_xuat_hien = set()
    duong_di_moi = []
   
    for diem in duong_di:
        if diem not in da_xuat_hien:
            duong_di_moi.append(diem)
            da_xuat_hien.add(diem)
   
    # Đảm bảo kết thúc tại đích
    if duong_di_moi[-1] != diem_ket_thuc:
        # Tìm đường từ điểm cuối đến đích
        duong_di_moi = _noi_den_dich(duong_di_moi, do_thi, diem_ket_thuc)
   
    return duong_di_moi


def _noi_den_dich(duong_di, do_thi, diem_ket_thuc):
    #Nối đường đi hiện tại đến điểm kết thúc
    if not duong_di:
        return duong_di
       
    diem_hien_tai = duong_di[-1]
   
    # Nếu đã đến đích
    if diem_hien_tai == diem_ket_thuc:
        return duong_di
   
    # Thử đi thẳng đến đích
    if diem_ket_thuc in do_thi.get(diem_hien_tai, {}):
        return duong_di + [diem_ket_thuc]
   
    # Tìm đường ngắn từ điểm hiện tại đến đích
    for _ in range(len(do_thi)):
        cac_lua_chon = list(do_thi.get(diem_hien_tai, {}).keys())
        if not cac_lua_chon:
            break
           
        diem_tiep_theo = random.choice(cac_lua_chon)
        if diem_tiep_theo not in duong_di:  # Tránh vòng lặp
            duong_di.append(diem_tiep_theo)
            diem_hien_tai = diem_tiep_theo
           
            if diem_hien_tai == diem_ket_thuc:
                break
   
    return duong_di

# HÀM 4: Lai ghép
def lai_ghep(quan_the, do_thi, diem_bat_dau, diem_ket_thuc, ti_le_lai_ghep=0.8):
    #Lai ghép các cặp cha mẹ để tạo ra thế hệ con
    quan_the_moi = []
   
    # Xáo trộn ngẫu nhiên
    random.shuffle(quan_the)
   
    for i in range(0, len(quan_the), 2):
        if i + 1 < len(quan_the) and random.random() < ti_le_lai_ghep:
            cha = quan_the[i]
            me = quan_the[i + 1]
           
            con_1, con_2 = _lai_ghep_cap(cha, me, do_thi, diem_bat_dau, diem_ket_thuc)
            quan_the_moi.extend([con_1, con_2])
        else:
            # Giữ nguyên nếu không lai ghép
            if i < len(quan_the):
                quan_the_moi.append(quan_the[i])
            if i + 1 < len(quan_the):
                quan_the_moi.append(quan_the[i + 1])
   
    return quan_the_moi


def _lai_ghep_cap(duong_di_1, duong_di_2, do_thi, diem_bat_dau, diem_ket_thuc):
    #Lai ghép một cặp đường đi
    if len(duong_di_1) < 3 or len(duong_di_2) < 3:
        return duong_di_1, duong_di_2
   
    # Chọn điểm cắt ngẫu nhiên (không phải điểm đầu/cuối)
    diem_cat_1 = random.randint(1, len(duong_di_1) - 2)
    diem_cat_2 = random.randint(1, len(duong_di_2) - 2)
   
    # Tạo con 1: phần đầu của cha + phần cuối của mẹ
    con_1 = duong_di_1[:diem_cat_1]
    for diem in duong_di_2:
        if diem not in con_1:
            con_1.append(diem)
       
# Ví dụ: Duyệt qua duong_di_2: ['F', 'E', 'D', 'C', 'B', 'A']
# 'F' not in ['A','B','C']? okey → thêm → ['A','B','C','F']
# 'E' not in ['A','B','C','F']? okey → thêm → ['A','B','C','F','E']  
# 'D' not in ['A','B','C','F','E']? okey → thêm → ['A','B','C','F','E','D']
# 'C' not in ['A','B','C','F','E','D']? có ròi → bỏ qua
# ...
    # Tạo con 2: phần đầu của mẹ + phần cuối của cha  
    con_2 = duong_di_2[:diem_cat_2]
    for diem in duong_di_1:
        if diem not in con_2:
            con_2.append(diem)
   
    # Sửa chữa đường đi để hợp lệ
    con_1 = _sua_duong_di(con_1, do_thi, diem_bat_dau, diem_ket_thuc)
    con_2 = _sua_duong_di(con_2, do_thi, diem_bat_dau, diem_ket_thuc)
   
    return con_1, con_2





# HÀM 5: Đột biến
def dot_bien(quan_the, do_thi, ti_le_dot_bien, diem_bat_dau, diem_ket_thuc):
    #Áp dụng đột biến cho quần thể
    quan_the_moi = []
   
    for duong_di in quan_the:
        if random.random() < ti_le_dot_bien:
            duong_di_da_dot_bien = _dot_bien_duong_di(duong_di, do_thi, diem_bat_dau, diem_ket_thuc)
            quan_the_moi.append(duong_di_da_dot_bien)
        else:
            quan_the_moi.append(duong_di)
   
    return quan_the_moi


def _dot_bien_duong_di(duong_di, do_thi, diem_bat_dau, diem_ket_thuc):
    #Đột biến một đường đi bằng cách thay đổi ngẫu nhiên 1 node trung gian
    if len(duong_di) <= 3:  # Quá ngắn để đột biến
        return duong_di
   
    # Chọn vị trí đột biến (không phải điểm đầu/cuối)
    vi_tri_dot_bien = random.randint(1, len(duong_di) - 2)
   
    diem_truoc = duong_di[vi_tri_dot_bien - 1]
    diem_sau = duong_di[vi_tri_dot_bien + 1]
   
    # Tìm các điểm có thể thay thế
    cac_diem_thay_the = []
    for diem_lan_can in do_thi.get(diem_truoc, {}):
        if (diem_lan_can in do_thi and        #điểm lận cận phải là một điểm trong đồ thị
            diem_sau in do_thi[diem_lan_can] and # điểm sau thuộc danh sách điểm lân cận nằm trong đồ thị các điểm lân cận
            diem_lan_can not in duong_di[:vi_tri_dot_bien] and # điểm lân cận không nằm trong khoảng từ đầu đến vị trí đột biến để tránh mắc chu trình và bị lặp
            diem_lan_can != diem_sau): # điểm lân cận không được trùng với điểm sau ví dụ duong_di_moi = ['A', 'B', 'D', 'D']
            cac_diem_thay_the.append(diem_lan_can)
   
    if cac_diem_thay_the:
        diem_moi = random.choice(cac_diem_thay_the)
        duong_di_moi = (duong_di[:vi_tri_dot_bien] +
                        [diem_moi] +
                        duong_di[vi_tri_dot_bien + 1:])
        return _sua_duong_di(duong_di_moi, do_thi, diem_bat_dau, diem_ket_thuc)
    # phải sửa lại đương đi bởi vì có thể sau khi đột biến nó không còn hợp lệ nữa
    # ( hai đỉnh đó chưa chưa chắc có nối nhau được không vì một trong hai đã bị đột biến thành đỉnh mới)
    #đồng thời xem đột biến xong có gây trùng lặp không
    # có khi đột biến sẽ gây mất đỉnh kết thúc cũng phải sửa để nối đến đích
   
    return duong_di

# HÀM 6: Thuật toán GA chính
def giai_thuat_GA(do_thi, diem_bat_dau, diem_ket_thuc,
                 kich_thuoc_quan_the=50, so_the_he_toi_da=100,
                 ti_le_dot_bien=0.1, ti_le_lai_ghep=0.8):
    """
    Giải thuật Di Truyền cho bài toán tìm đường đi tối ưu
    """
    # Bước 1: Khởi tạo quần thể
    quan_the = khoi_tao_quan_the(do_thi, diem_bat_dau, diem_ket_thuc, kich_thuoc_quan_the)
   
    duong_di_tot_nhat = None
    fitness_tot_nhat = 0
   
    for the_he in range(so_the_he_toi_da):
        # Bước 2: Tính fitness
        danh_sach_fitness = tinh_fitness(quan_the, do_thi)
       
        # Cập nhật giải pháp tốt nhất
        chi_so_tot_nhat = np.argmax(danh_sach_fitness)
        fitness_hien_tai = danh_sach_fitness[chi_so_tot_nhat]
       
        if fitness_hien_tai > fitness_tot_nhat:
            fitness_tot_nhat = fitness_hien_tai
            duong_di_tot_nhat = quan_the[chi_so_tot_nhat]
            print(f"Thế hệ {the_he}: Tìm thấy đường đi tốt hơn, chi phí = {1/fitness_tot_nhat:.2f}")
       
        # Bước 3: Chọn lọc
        quan_the = chon_loc(quan_the, danh_sach_fitness, 'tournament')
       
        # Bước 4: Lai ghép
        quan_the = lai_ghep(quan_the, do_thi, diem_bat_dau, diem_ket_thuc, ti_le_lai_ghep)
       
        # Bước 5: Đột biến
        quan_the = dot_bien(quan_the, do_thi, ti_le_dot_bien, diem_bat_dau, diem_ket_thuc)
       
        # Elitism: Giữ lại cá thể tốt nhất
        if duong_di_tot_nhat and len(quan_the) > 0:
            quan_the[0] = duong_di_tot_nhat
   
    # Trả về kết quả tốt nhất
    chi_phi_tot_nhat = 1.0 / fitness_tot_nhat if fitness_tot_nhat > 0 else float('inf')
    return duong_di_tot_nhat, chi_phi_tot_nhat


do_thi_giao_thong = {
        'A': {'B': 4, 'C': 2, 'D': 5},
        'B': {'A': 4, 'C': 1, 'E': 3, 'F': 8},
        'C': {'A': 2, 'B': 1, 'D': 3, 'E': 2},
        'D': {'A': 5, 'C': 3, 'E': 4, 'G': 6},
        'E': {'B': 3, 'C': 2, 'D': 4, 'F': 5, 'G': 3},
        'F': {'B': 8, 'E': 5, 'G': 2, 'H': 7},
        'G': {'D': 6, 'E': 3, 'F': 2, 'H': 4},
        'H': {'F': 7, 'G': 4}
    }

 # Test 1: Khởi tạo quần thể
print("1. KIỂM TRA KHỞI TẠO QUẦN THỂ:")
quan_the = khoi_tao_quan_the(do_thi_giao_thong, 'A', 'H', 5)
print(f"   Số lượng cá thể: {len(quan_the)}")
for i, duong_di in enumerate(quan_the):
        print(f"   Cá thể {i+1}: {' -> '.join(duong_di)}")

# Test 2: Tính fitness
print("\n2. KIỂM TRA TÍNH FITNESS:")
danh_sach_fitness = tinh_fitness(quan_the, do_thi_giao_thong)
for i, (duong_di, fitness) in enumerate(zip(quan_the, danh_sach_fitness)):
        chi_phi = 1/fitness if fitness > 0 else "Vô cực"
        print(f"   Cá thể {i+1}: {duong_di} -> Fitness: {fitness:.4f}, Chi phí: {chi_phi}")
   

# Test 3: Chọn lọc
print("\n3. KIỂM TRA CHỌN LỌC:")
quan_the_chon_loc = chon_loc(quan_the, danh_sach_fitness, 'tournament')
print(f"   Sau chọn lọc: {len(quan_the_chon_loc)} cá thể")

# Test 4: Lai ghép
print("\n4. KIỂM TRA LAI GHÉP:")
quan_the_lai_ghep = lai_ghep(quan_the_chon_loc, do_thi_giao_thong, 'A', 'H', 0.8)
print(f"   Sau lai ghép: {len(quan_the_lai_ghep)} cá thể")
for i, duong_di in enumerate(quan_the_lai_ghep[:3]):  # Hiển thị 3 cá thể đầu
        print(f"   Cá thể {i+1}: {' -> '.join(duong_di)}")

# Test 5: Đột biến
print("\n5. KIỂM TRA ĐỘT BIẾN:")
quan_the_dot_bien = dot_bien(quan_the_lai_ghep, do_thi_giao_thong, 0.3, 'A', 'H')
print(f"   Sau đột biến: {len(quan_the_dot_bien)} cá thể")

# Test 6: Chạy toàn bộ thuật toán GA
print("\n6. THUẬT TOÁN GA:")
duong_di_tot_nhat, chi_phi_tot_nhat = giai_thuat_GA(
        do_thi=do_thi_giao_thong,
        diem_bat_dau='A',
        diem_ket_thuc='H',
        kich_thuoc_quan_the=20,
        so_the_he_toi_da=50,
        ti_le_dot_bien=0.15,
        ti_le_lai_ghep=0.8
    )
print(f"Đường đi tối ưu: {' -> '.join(duong_di_tot_nhat)}")
print(f"Tổng chi phí: {chi_phi_tot_nhat}")
print(f"Số điểm dừng: {len(duong_di_tot_nhat)}")

if _kiem_tra_duong_di_hop_le(duong_di_tot_nhat, do_thi_giao_thong):
        print("Đường đi hợp lệ")
else:
        print("Đường đi không hợp lệ")
