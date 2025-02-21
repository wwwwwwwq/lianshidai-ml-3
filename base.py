# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
plt.rcParams['mathtext.fontset'] ='cm'#设置 LaTeX 字体

#加载和预处理数据函数
def load_process_data():
    #从expolanet_data.csv文件（当前目录）中读取数据
    #由于读取的是DataFrame类型，所以这里用df缩写表示读取的数据
    #用pandas库来读取csv文件（这里简写为pd）
    df=pd.read_csv('exoplanet_data.csv',comment='#')

    #自变量（输入变量）x，其值为轨道半长轴a和恒星指令m
    x=df[['pl_orbsmax','st_mass']]

    #因变量（输出变量）y，其值为轨道周期
    y=df['pl_orbper']

    #删除缺失值。
    #因为线性回归算法不能处理缺失值（NaN和null），所以要手动处理缺失值，防止数据中有缺失值出现。
    
    data=pd.concat([x,y],axis=1)#pandas的连接函数，axis=1表示水平连接，所以这里将x和y水平连接起来
    #举个例子，比如：
    """
    原始数据：
    x:
       pl_orbsmax  st_mass
    0    1.0        2.0
    1    2.0        NaN#缺失值
    2    3.0        3.0
    3    4.0        4.0

    y:
    0    10.0
    1    20.0
    2    NaN#缺失值
    3    40.0

    连接后：
       pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    1    2.0        NaN      20.0    # 有缺失值
    2    3.0        3.0      NaN     # 有缺失值
    3    4.0        4.0      40.0
    """

    data=data.dropna()#删除包含缺失值的行
    #删除缺失值后，数据如下：
    """
    pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    3    4.0        4.0      40.0
    """

    #接下来就是从删除后的数据中提取自变量x和因变量y
    x=data[['pl_orbsmax','st_mass']]
    y=data['pl_orbper']

    #那么处理后x和y就变成了：
    """
    x:
   pl_orbsmax  st_mass
    0    1.0        2.0
    3    4.0        4.0

    y:
    0    10.0
    3    40.0
    """

    #返回处理后的自变量x和因变量y
    return x,y

#训练和测试模型函数
def train_test_model(x,y,test_size=0.7):
    #将数据集划分为训练集和测试集

    #test_size表示有多少的数据用来测试，random_state=42表示随机种子为42。随机种子的设置是为了每次运行代码时，划分结果都相同。
    #这里选择随机种子为42是因为42是宇宙的终极奥秘（）。
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)

    #创建并训练线性回归模型

    #这里利用scikit-learn库中的LinearRegression类来创建线性回归模型，其中model就是LinearRegression类的一个实例。
    #LinearRegression类是scikit-learn库中的一个线性回归类，用于构建类似于y=ax_{1}+bx_{2}+c的方程。其中y代表我们要预测的轨道周期，x_{1}代表轨道半长轴，x_{2}代表恒星质量。
    model=LinearRegression()

    #利用LinearRegression类中的fit方法训练模型
    model.fit(x_train,y_train)

    #用LinearRegression类中的predict方法进行预测
    y_pred=model.predict(x_test)

    #利用mean_squared_error函数计算均方误差
    mse=mean_squared_error(y_test,y_pred)

    #返回model,mse,y_test,y_pred
    return model,mse,y_test,y_pred

#手动实现最小二乘法实现线性回归（Build it from scratch应该是这个意思吧（））
def mannual_linear_regression(x,y,test_size=0.7):
    #将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    #添加偏置项（也就是截距项），把矩阵x的最左边那一行全部变成1，以便于出现常数项
    #比如：
    """
    原始数据：
    x:
    |2 3|
    |4 5|
    样本：
    x1=2,x2=3
    x1=4,x2=5

    添加偏置项后：
    x:
    |1 2 3|
    |1 4 5|

    就会是：
    result:
    |1xc 2xβ1 3xβ2|
    |1xc 4xβ1 5xβ2|
    """
    #添加偏置项
    x_train_b=np.c_[np.ones(x_train.shape[0]),x_train]

    #计算最小二乘法的解析解：β=(X^T*X)^-1*X^T*y
    results=np.linalg.inv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)

    #计算测试集的预测值
    x_test_b=np.c_[np.ones(x_test.shape[0]),x_test]
    y_pred=x_test_b.dot(results)

    #计算均方误差
    mse=mean_squared_error(y_test,y_pred)

    #返回结果
    #results[0]是截距项，results[1:]是系数项
    return results[0],results[1:],y_test,y_pred,mse

#利用plt可视化预测结果
def visualize_results(x,y,mode=1):
    #创建图形窗口，并将大小设置为10x6英寸
    fig=plt.figure(figsize=(10,6))

    #创建GridSpec对象，用于控制子图的布局
    #2,1表示创建一个2行1列的子图
    #height_ratios=[1,4]表示第一行的高度为1，第二行的高度为4
    #hspace=0.2表示子图之间的垂直间距为0.2
    gs=fig.add_gridspec(2,1,height_ratios=[1,4],hspace=0.2)

    #在网格的第一行创建一个子图，用于显示线性回归方程，评估指标
    #gs[0]表示网格的第一行
    ax_text=fig.add_subplot(gs[0])

    #在网格的第二行创建一个散点图子图
    #gs[1]表示网格的第二行
    ax_scatter=fig.add_subplot(gs[1])

    #调整子图布局，在底部留出0.2的空白区域放置滑动条
    plt.subplots_adjust(bottom=0.2)
    
    if mode==1:
        #获取train_test_model的返回值
        model,mse,y_test,y_pred=train_test_model(x,y,test_size=0.7)

        #获取线性回归方程的文本
        equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
    elif mode==2:
        #获取mannual_linear_regression的返回值
        intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=0.7)

        #获取线性回归方程的文本
        equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'

    #获取MSE文本
    mse_text=f'均方误差（$MSE$）：{mse:.3f}'

    #plt.scatter()来绘制散点图，其中第一个参数是x轴（表示真实值），第二个参数是y轴（表示预测值），第三个参数是透明度
    scatter=ax_scatter.scatter(y_test,y_pred,alpha=0.5,label='pred vs. true')

    #获取坐标轴的最小值和最大值
    ax_min=min(y_test.min(),y_pred.min())
    ax_max=max(y_test.max(),y_pred.max())

    #绘制理想的预测线（ideal），也就是对角线
    #[y_test.min(),y_test.max()]：x轴的起点和终点
    #[y_test.min(),y_test.max()]：y轴的起点和终点
    #'r--'：红色虚线（r代表red红色，--代表虚线）
    #lw=2：线宽为2
    line,=ax_scatter.plot([ax_min,ax_max],[ax_min,ax_max],'r--',lw=2,label='ideal')

    #设置散点图的坐标轴范围，使之变成一个正方形
    ax_scatter.set_xlim(ax_min,ax_max)
    ax_scatter.set_ylim(ax_min,ax_max)
    #设置散点图的坐标轴比例
    ax_scatter.set_aspect('equal',adjustable='box')
        
    #创建一个滑动条，用于控制测试集的比例test_size
    #0.2：滑动条的x轴
    #0.1：滑动条的y轴
    #0.6：滑动条的宽度
    #0.03：滑动条的高度
    ax_slider=plt.axes([0.2,0.1,0.6,0.03])
    #创建一个调节范围为0-1的滑动条，并且初始值为0.7
    slider=Slider(ax_slider,'Test Size',0,1,valinit=0.7)

    #文本信息关闭坐标轴
    ax_text.axis('off')    

    #添加到ax_text中
    #0.05代表文本框左上角的x坐标
    #0.5代表文本框左上角的y坐标
    #transform指定坐标系是transAxes(相对坐标系)，坐标范围为[0,1]
    #verticalalignment='center'表示文本垂直居中于(0.05,0.5)
    #bbox=dict()设置文本框的背景和边框样式，boxstyle='round'文本框为圆角矩形，facecolor='white'背景颜色为白色,alpha=0.8透明度为0.8
    ax_text.text(0.05,0.5,f'线性回归方程：\n{equ}\n\n评估指标：\n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    #利用滑动条中的值来更新test_size
    def update(val):
        #获取滑动条中的值并进行赋值
        test_size=slider.val
        if mode==1:
            #重新利用test_size获取测试数据
            model,mse,y_test,y_pred=train_test_model(x,y,test_size=test_size)
            #获取线性回归方程的文本
            equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
        elif mode==2:
            #重新利用test_size获取测试数据
            intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=test_size)
            #获取线性回归方程的文本
            equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'
            
        #获取MSE文本
        mse_text=f'均方误差（$MSE$）：{mse:.3f}'

        #更新散点图
        scatter.set_offsets(np.c_[y_test,y_pred])

        #更新文本注释
        #先清理原本的数据
        ax_text.clear()
        #关闭坐标轴
        ax_text.axis('off')
        #添加新的文本注释
        ax_text.text(0.05,0.5,f'线性回归方程： \n{equ}\n\n评估指标： \n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        #更新坐标轴的最大值和最小值
        ax_min=min(y_test.min(),y_pred.min())
        ax_max=max(y_test.max(),y_pred.max())

        #更新理想预测线
        line.set_data([ax_min,ax_max],[ax_min,ax_max])

        #更新坐标轴的范围
        ax_scatter.set_xlim(ax_min,ax_max)
        ax_scatter.set_ylim(ax_min,ax_max)
        #更新坐标轴的比例
        ax_scatter.set_aspect('equal',adjustable='box')

        #更新图表，这里利用draw_idle()不用draw()是因为draw_idle()更适合交互式的东西比如滑动条
        fig.canvas.draw_idle()

    #利用滑动条的on_changed方法来更新图表
    slider.on_changed(update)

    #在左上角添加图例
    ax_scatter.legend(loc='upper left')

    #设置x轴标签（示意图中给的是T_true）
    ax_scatter.set_xlabel('T_true')

    #设置y轴标签（示意图中给的是T_pred）
    ax_scatter.set_ylabel('T_pred')

    #设置图表的标题（示意图中给的是true vs. pred T）
    ax_scatter.set_title('true vs. pred T')

    ax_scatter.grid(True)

    #显示图表
    plt.show()

#主函数
def main(mode=1):
    #加载和预处理数据
    x,y=load_process_data()

    #可视化预测结果
    visualize_results(x,y,mode)

#如果当前文件是主程序，而不是被用作模块，则执行main函数
if __name__=='__main__':
    if len(sys.argv)>=2:
        main(int(sys.argv[1]))

