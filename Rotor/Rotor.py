import matplotlib.pyplot as plt   # Vẽ đồ thị
import numpy                     # Các phép toán ma trận
import math                      # Các phép toán toán học
import yaml                      # Làm việc với file YAML
import mujoco                    # Thư viện MuJoCo
import numpy                     # Thư viện NumPy
import mediapy as media          # Thư viện cho media (video, ảnh)
import os
import imageio
os.environ['FFMPEG_BINARY'] = "ffmpeg.exe"



ts = 1e-4                    # Timestep cho mô phỏng Mujoco (s)
render_width = 800
render_height = 608 
xml_template = """
<mujoco>
    <visual>
        <global offwidth="512" offheight="512"/>
    </visual>
    <option timestep="{ts}"/>
    <asset>
		<material name="floor" texture="check1" texrepeat="2 2" texuniform="true"/>
		<texture name="check1" builtin="checker" type="2d" width="256" height="256" rgb1="1 1 1" rgb2="0 0 0"/>

		<material name="object" texture="check2" texrepeat="5 5" texuniform="true"/>
		<texture name="check2" builtin="checker" type="2d" width="256" height="256" rgb1="0 0 1" rgb2="0 1 0"/>
	</asset>
  <worldbody>
		<light diffuse=".5 .5 .5" pos="0 0 12" dir="0 0 -1"/>
		

		
	
			
 <!--2 rotor quay-->
 <body pos="0 0 0">
	<joint type="hinge" name="ball" axis="0 0 1"/>
    <geom type="sphere" mass=".15" material="object" size=".0025"/> <!--cụm roto là tổng hợp của nhiều thứ liền kề, tong 0.7 [2]-->
			
			<!--khoi luong moi rotor la 0.012 nhu da ghi-->
			<body pos="0 -.014 .005">
				<joint type="hinge" name="R1" axis="0 0 1" pos="0 .014 0"/>
				<geom type="box" mass="0.012" size=".0025 .014 .00125" rgba="1 0 0 1"/>
			</body>

			<body pos="0 -.014 -.005">
				<joint type="hinge" name="R2" axis="0 0 -1" pos="0 .014 0"/>
				<geom type="box" mass="0.012" size=".0025 .014 .00125" rgba="0 1 0 1"/>
			</body>

</body>
			

	</worldbody>

	<actuator>
		<motor joint="R1" gear="0.1"/>
		<motor joint="R2" gear="0.1"/>
	
	</actuator>
	


</mujoco>
"""
xml = xml_template.format(ts=ts, render_width=320, render_height=240)

#Tạo mô hình Mujoco, khởi tạo dữ liệu và renderer
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, width=416, height=304)
#Tính toán các thông số thời gian cho mô phỏng
duration =  3      # Thời gian mô phỏng (s)
framerate = 20               # Tốc độ khung hình hiển thị (Hz)
data_rate = 30.04502260677355     # Tần số lấy mẫu dữ liệu thực nghiệm (Hz)
#print(duration, data_rate)   # => 7.122644 30.04502260677355



def run_sim(render=False, video_filename=None):
    V_supply = 6  # điện áp cung cấp trong thí nghiệm
    mujoco.mj_resetData(model, data)       # reset trước
    data.qpos[0] = numpy.pi / 6            # đặt lại góc sau reset
    mujoco.mj_forward(model, data)         # cập nhật trạng thái mới
    print (data.qpos[0])

    m = 0.012   # khối lượng rotor (kg)
    r = 0.014   # bán kính lệch tâm (m)
    f = 30      # tần số quay mong muốn (Hz)
    omega_d = 2 * numpy.pi * f  # tốc độ góc mục tiêu (rad/s)
    I = m * r**2  # mômen quán tính xấp xỉ của rotor (kg·m²)

    Kp = 0.0001    # hệ số tỉ lệ (tạo torque khi có sai số tốc độ)
    Kd = 0.001  # hệ số dập tắt (tạo torque chống rung, chống overshoot)

    def mycontroller(model, data):
        w1 = data.qvel[1]   # tốc độ góc joint R1
        w2 = data.qvel[2]   # tốc độ góc joint R2
        err1 = omega_d - w1
        err2 = omega_d - w2

        #công thức PD: t = Kp * (W_desired-W_sim) - Kd*W_sim
        torque1 = Kp * err1 - Kd * w1
        torque2 = Kp * err2 - Kd * w2

        data.ctrl[0] = torque1
        data.ctrl[1] = torque2
     
        #Hàm điều khiển này tính toán mô-men xoắn cần thiết cho khớp joint_1 dựa trên lệnh điều khiển theo thời gian và trạng thái hiện tại của servo.
        



        #Xác định tín hiệu điều khiển mong muốn
        
    try:
        mujoco.set_mjcb_control(mycontroller)
        q, w, t_arr = [], [], []

        # mở writer nếu cần lưu video
        if video_filename is not None:
            writer = imageio.get_writer(video_filename, fps=framerate, macro_block_size=1,codec='mpeg4')

        frame_count = 0
        while data.time < duration:
            mujoco.mj_step(model, data)

            # render theo framerate
            if render and frame_count < data.time * framerate:
                renderer.update_scene(data)
                pixels = renderer.render()
                if video_filename is not None:
                    writer.append_data(pixels)
                frame_count += 1

            # lưu dữ liệu
            if len(t_arr) < data.time * data_rate:
                q.append(data.qpos.copy())
                w.append(data.qvel.copy())
                t_arr.append(data.time)

        if video_filename is not None:
            writer.close()

        mujoco.set_mjcb_control(None)
        return numpy.array(t_arr), numpy.array(q)



    except Exception as ex:
        mujoco.set_mjcb_control(None)
        raise
run_sim(render=True,video_filename='Output6.mp4')
renderer.close()










# import matplotlib.pyplot as plt   # Vẽ đồ thị
# import numpy                     # Các phép toán ma trận
# import math                      # Các phép toán toán học
# import yaml                      # Làm việc với file YAML
# import mujoco                    # Thư viện MuJoCo
# import numpy                     # Thư viện NumPy
# import mediapy as media          # Thư viện cho media (video, ảnh)
# import os
# os.environ['FFMPEG_BINARY'] = "ffmpeg.exe"

# #Ham song vuong
# def square(t, A, f, w, b, t_0):
#     y = (t - t_0) * f         # Dịch thời gian theo tần số
#     y = y % 1                 # Lấy phần dư để đảm bảo sóng có chu kỳ
#     y = (y < w) * 1           # Xác định nếu sóng ở mức cao (tín hiệu dương)
#     y = A * y + b             # Áp dụng biên độ và độ lệch
#     return y

# Vnom = 6                     # Điện áp danh định của servo (V)
# G = 55.5                     # Tỉ số truyền của hộp số
# t_stall = 0.15 / G           # Mô-men cực đại tại trạng thái dừng (Nm)
# i_stall = 0.6                # Dòng điện cực đại tại trạng thái dừng (A)
# R = Vnom / i_stall           # Điện trở cuộn dây động cơ (Ohm)
# i_nl = 0.2                   # Dòng không tải (A)
# w_nl = 0.66 * 1000 * math.pi / 180 * G  # Tốc độ góc không tải (rad/s)
# kt = t_stall / i_stall       # Hằng số mô-men xoắn (Nm/A)
# ke = kt                      # Hằng số phản điện động (V·s/rad) ~ bằng kt trong SI
# b = kt * i_nl / w_nl         # Hệ số giảm chấn nhớt do ma sát nội bộ
# ts = 1e-4                    # Timestep cho mô phỏng Mujoco (s)

# with open('du_lieu_servo_thu_duoc.yml') as f:
#     servo_data = yaml.load(f, Loader=yaml.Loader)

# t_data = numpy.array(servo_data['t'])       # Dữ liệu thời gian (s)
# print(t_data.shape)                                # Kết quả: (214,)
# dt_data = (t_data[-1] - t_data[0]) / len(t_data)  # Khoảng thời gian giữa các mẫu (s)
# q_data = numpy.array([servo_data['theta_u']]).T  # Dữ liệu góc quay (rad) #(1, 214)
# print(q_data.shape)                                # Kết quả: (214, 1)

# #ham song vuong
# desired = square(
#     t_data,
#     A = servo_data['A'],
#     f = servo_data['f'],
#     w = servo_data['w'],
#     b = servo_data['b'],
#     t_0 = servo_data['t_0']
# )

# render_width = 800
# render_height = 600
# xml_template = """
# <mujoco>
#     <visual><global offwidth="{render_width}" offheight="{render_height}" /></visual>
#     <option timestep="{ts}"/>
#     <asset>
# 		<material name="floor" texture="check1" texrepeat="2 2" texuniform="true"/>
# 		<texture name="check1" builtin="checker" type="2d" width="256" height="256" rgb1="1 1 1" rgb2="0 0 0"/>

# 		<material name="object" texture="check2" texrepeat="5 5" texuniform="true"/>
# 		<texture name="check2" builtin="checker" type="2d" width="256" height="256" rgb1="0 0 1" rgb2="0 1 0"/>
# 	</asset>
#   <worldbody>
# 		<light diffuse=".5 .5 .5" pos="0 0 12" dir="0 0 -1"/>
		

		
	
			
#  <!--2 rotor quay-->
#  <body pos="0 0 0">
# 	<joint type="hinge" name="ball" axis="0 0 1"/>
#     <geom type="sphere" mass=".15" material="object" size=".0025"/> <!--cụm roto là tổng hợp của nhiều thứ liền kề, tong 0.7 [2]-->
			
# 			<!--khoi luong moi rotor la 0.012 nhu da ghi-->
# 			<body pos="0 -.014 .005">
# 				<joint type="hinge" name="R1" axis="0 0 1" pos="0 .014 0"/>
# 				<geom type="box" mass="0.012" size=".0025 .014 .00125" rgba="1 0 0 1"/>
# 			</body>

# 			<body pos="0 -.014 -.005">
# 				<joint type="hinge" name="R2" axis="0 0 -1" pos="0 .014 0"/>
# 				<geom type="box" mass="0.012" size=".0025 .014 .00125" rgba="0 1 0 1"/>
# 			</body>

# </body>
			

# 	</worldbody>

# 	<actuator>
# 		<motor joint="R1" gear="0.1"/>
# 		<motor joint="R2" gear="0.1"/>
	
# 	</actuator>
	


# </mujoco>
# """
# xml = xml_template.format(ts=ts, render_width=render_width, render_height=render_height)

# #Tạo mô hình Mujoco, khởi tạo dữ liệu và renderer
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model, width=render_width, height=render_height)
# #Tính toán các thông số thời gian cho mô phỏng
# duration = t_data[-1]        # Thời gian mô phỏng (s)
# framerate = 30               # Tốc độ khung hình hiển thị (Hz)
# data_rate = 1 / dt_data      # Tần số lấy mẫu dữ liệu thực nghiệm (Hz)
# print(duration, data_rate)   # => 7.122644 30.04502260677355



# def run_sim(kp, b_act, render=False, video_filename=None):
#     V_supply = 6  # điện áp cung cấp trong thí nghiệm
#     mujoco.mj_resetData(model, data)       # reset trước
#     data.qpos[0] = numpy.pi / 6            # đặt lại góc sau reset
#     mujoco.mj_forward(model, data)         # cập nhật trạng thái mới
#     print (data.qpos[0])
#     def mycontroller1(model, data):
     
#         #Hàm điều khiển này tính toán mô-men xoắn cần thiết cho khớp joint_1 dựa trên lệnh điều khiển theo thời gian và trạng thái hiện tại của servo.
        
#         w = data.qvel[1]       # vận tốc góc hiện tại
#         actual = data.qpos[1]  # vị trí thực tế hiện tại
#         t = data.time          # thời gian mô phỏng hiện tại

#         #Xác định tín hiệu điều khiển mong muốn
#         desired = square(t, A=math.radians(180), f=100, w=0.5, b=0, t_0=0.5015946)  # vị trí mong muốn

#         error = desired - actual # sai số vị trí
#         V = kp * error # Nếu sai số lớn, thì điện áp điều khiển V lớn đẩy motor quay nhanh hơn.

#         # Giới hạn điện áp trong khoảng ±V_supply
#         if V > V_supply: V = V_supply
#         if V < -V_supply: V = -V_supply

#         # Tính mô-men xoắn và gán cho khớp
#         torque = (kt * (V - ke * w * G) / R - b_act * w * G) * G
#         data.ctrl[0] = torque
#         data.ctrl[1] = torque
#         return
 
#     try:
#         mujoco.set_mjcb_control(mycontroller1)
#         q = []
#         w = []
#         t = []
#         frames = []
#         while data.time < duration:
#             # print(data.time)
#             mujoco.mj_step(model, data)
#             if len(frames) < data.time * framerate:
#                 renderer.update_scene(data)
#                 pixels = renderer.render()
#                 frames.append(pixels)
#             if len(t) < data.time * data_rate:
#                 # print(data.time)
#                 q.append(data.qpos.copy())
#                 w.append(data.qvel.copy())
#                 t.append(data.time)
#         if render:
#             media.show_video(frames, fps=framerate,width=render_width,height = render_height)
#             if video_filename is not None:
#                 media.write_video(video_filename,frames,fps=framerate)
#         mujoco.set_mjcb_control(None)
#         t = numpy.array(t)
#         q = numpy.array(q)
#         q = q[:len(q_data)]
#     except Exception as ex:
#         mujoco.set_mjcb_control(None)
#         raise

#     return t, q

# t,q = run_sim(kp = 8.896713022899348,b_act = 1.4041787991974594e-06,render=True,video_filename='output4.mp4')
# # plt.figure() # ve bieu do
# # a2 = plt.plot(t_data, desired)   # Tín hiệu điều khiển mong muốn
# # a3 = plt.plot(t_data, q_data)    # Dữ liệu thực tế (từ cảm biến)
# # a1 = plt.plot(t_data, q)         # Kết quả mô phỏng
# # plt.legend(a1 + a2 + a3, ['sim', 'control', 'actual'])

# # #Uoc luong kp (V = kp*error) va b
# # t,q1 = run_sim(kp = 2,b_act = b,render=False)
# # t,q2 = run_sim(kp = 4,b_act = b,render=False)
# # t,q3 = run_sim(kp = 20,b_act = b,render=False)
# # plt.figure() #Tham so kp
# # a1 = plt.plot(t_data,q1)
# # a2  = plt.plot(t_data,q2)
# # a3 = plt.plot(t_data,q3)
# # plt.legend(a1+a2+a3,['$k_p=2$','$k_p=4$','$k_p=20$'])

# # t,q1 = run_sim(kp = 20, b_act = b, render=False)
# # t,q2 = run_sim(kp = 20, b_act = b*3, render=False)
# # t,q3 = run_sim(kp = 20, b_act = b*6, render=False)

# # plt.figure() #Tham so b
# # a1 = plt.plot(t_data,q1)
# # a2 = plt.plot(t_data,q2)
# # a3 = plt.plot(t_data,q3)
# # plt.legend(a1+a2+a3,['$b={b:1.2e}$'.format(b=b),'$b={b:1.2e}$'.format(b=b*3),'$b={b:1.2e}$'.format(b=b*6)])


# # #Xac dinh tham son kp va b
# # import scipy.optimize as so
# # def fun(vars):
# #     k, b = vars
# #     t, q = run_sim(k, b) #time and position
# #     error = q - q_data
# #     error = error**2
# #     error = error.sum()
# #     error = error**0.5 # tính sai số RMS (căn tổng bình phương sai số)
# #     print(k, b, error)
# #     return error
# # #Sai số từ dự đoán khoảng 1.4589710236248898. 
# # print("Sai so voi tham so ban dau")
# # ini = [15, b]
# # print(fun(ini))
# # print("Tim tham so toi uu")
# # results = so.minimize(fun, x0=ini, method='nelder-mead',bounds=((1, 100), (b*0.1, b*10)), options={'xatol':1e-2, 'fatol':1e-2})
# # print(results)
# # # method='nelder-mead': thuật toán tối ưu được sử dụng
# # #bounds: giới hạn giá trị cho k (1 → 100) và b (0.1b → 10b)
# # #xatol, fatol: độ chính xác cần đạt (nếu nhỏ hơn thì dừng)

# # #8.896713022899348 1.4041787991974594e-06 1.4356944007506427
# # t,q = run_sim(*results.x,render=True,video_filename='output2.mp4')
# # # # Và vẽ kết quả toi uu
# # # plt.figure()
# # # a2 = plt.plot(t_data,desired)
# # # a3  = plt.plot(t_data,q_data)
# # # a1 = plt.plot(t_data,q)
# # # plt.legend(a1+a2+a3,['sim','control','actual'])
# # # #plt.show()
# renderer.close()

