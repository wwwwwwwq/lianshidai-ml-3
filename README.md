<font color="#87CEEB" size="5">? ����ʼ�</font>

### д��ǰ��

<details>
<summary><strong>? д��ǰ��</strong></summary>

---

����Ŀ�Ѿ���Դ��github�ֿ⣺  
```bash
https://github.com/wwwwwwwq/lianshidai-ml-3.git
```

---

�ļ����¸���.py�ļ�˵����  
```bash
base.py���������֣�û�ж���ת����R����⣬����������ͼ��Ļ滭����Ĭ���ǽ���sklearnģ��ѵ��
base_mannual.py���ֶ�ʵ����С���˷�ʵ�����Իع飨Build it from scratchӦ���������˼��qwq��
base_sklearn.py��ʹ��sklearn.linear_model.LinearRegression��ѵ��ģ��

extend.py����չ���֣�������R�����Ͷ���ת����Ĭ���ǽ���sklearnģ��ѵ��
extend_mannual.py���ֶ�ʵ����С���˷�ʵ�����Իع�
extend_sklearn.py��ʹ��sklearn.linear_model.LinearRegression��ѵ��ģ��
```
  
����˵����  
```bash
python [base.py/extend.py] [1/2]
#[1/2]��ָ���������1������sklearn���������2������mannual

������ı��༭���������У���ô��Ӧ�������� .py�ļ�˵�� ����
```
  
��Ҫ�İ���
```bash
pandas,numpy,matplotlib,Scikit-Learn
```

---

**ע��**
�ڴ�ֻ�Ի������������⣬��չ����ֻ������������
</details>

### �������

<details>
<summary><strong>? �������</strong></summary>

---

��ž���˵����NASA ϵ���������ݿ�����ݣ��ѷ���exoplanet_data.csv�У���ȡ**����볤��(pl_orbsmax)**
��**ĸ��������(st_mass)**��������������Ϊ����������**�������(pl_orbper)** ��ΪĿ�������
���������Իع�ģ�͡�����Ҫ��**�������(MSE)** ������ģ�͵����ܡ�Ȼ��Ļ�����һ��**�ֶ�ʹ����С���˷�ʵ�����Իع�**������
</details>

### �������

<details>
<summary><strong>? ����</strong></summary>

---

**NASA ϵ���������ݿ��csv�ļ�**(exoplanet_data.csv)
</details>

<details>
<summary><strong>? ���</strong></summary>

---

**���ӻ�ɢ��ͼ**��**Ԥ��������**��**����ģ�͵����**
</details>

### �����Ŀ

<details>
<summary><strong>? ��������</strong></summary>

---

���ڽ��ܵ���.csv�ļ������ݣ�����Ϊ�˷����ֱ����pandas��read_csv������������
����Ҫע���������������exoplanet_data.csv�ļ��е���������#��ͷ�ģ�  
```bash
# This file was produced by the NASA Exoplanet Archive  http://exoplanetarchive.ipac.caltech.edu
# Wed Feb 19 21:07:49 2025
#
# COLUMN pl_name:        Planet Name
# COLUMN hostname:       Host Name
# COLUMN sy_snum:        Number of Stars
# COLUMN sy_pnum:        Number of Planets
#...
```
����˵������Ҫ������һ���֣�����read_csv�����һ�� **'comment='#''** �Ĳ�����  
```bash
    #��expolanet_data.csv�ļ�����ǰĿ¼���ж�ȡ����
    #���ڶ�ȡ����DataFrame���ͣ�����������df��д��ʾ��ȡ������
    #��pandas������ȡcsv�ļ��������дΪpd��
    df=pd.read_csv('exoplanet_data.csv',comment='#')
```
Ȼ��Ͷ�ȡ���������Ŀ�������  
```python
#�Ա��������������x����ֵΪ����볤��a�ͺ���ָ��m
    x=df[['pl_orbsmax','st_mass']]

    #����������������y����ֵΪ�������
    y=df['pl_orbper']
```
������Ҫ����ȱʧֵ��NaN����null������Ϊ���Իع��㷨���ܴ���ȱʧֵ����������Ӻ��õ��ĺ���������������ע�ͣ�  
```python
    #ɾ��ȱʧֵ��
    #��Ϊ���Իع��㷨���ܴ���ȱʧֵ��NaN��null��������Ҫ�ֶ�����ȱʧֵ����ֹ��������ȱʧֵ���֡�
    
    data=pd.concat([x,y],axis=1)#pandas�����Ӻ�����axis=1��ʾˮƽ���ӣ��������ｫx��yˮƽ��������
    #�ٸ����ӣ����磺
    """
    ԭʼ���ݣ�
    x:
       pl_orbsmax  st_mass
    0    1.0        2.0
    1    2.0        NaN#ȱʧֵ
    2    3.0        3.0
    3    4.0        4.0

    y:
    0    10.0
    1    20.0
    2    NaN#ȱʧֵ
    3    40.0

    ���Ӻ�
       pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    1    2.0        NaN      20.0    # ��ȱʧֵ
    2    3.0        3.0      NaN     # ��ȱʧֵ
    3    4.0        4.0      40.0
    """

    data=data.dropna()#ɾ������ȱʧֵ����
    #ɾ��ȱʧֵ���������£�
    """
    pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    3    4.0        4.0      40.0
    """

    #���������Ǵ�ɾ�������������ȡ�Ա���x�������y
    x=data[['pl_orbsmax','st_mass']]
    y=data['pl_orbper']

    #��ô�����x��y�ͱ���ˣ�
    """
    x:
   pl_orbsmax  st_mass
    0    1.0        2.0
    3    4.0        4.0

    y:
    0    10.0
    3    40.0
    """
```
����������֮�����ǾͿ���д�����غ�Ԥ������ **load_process_data** �ˣ�  
```python
#���غ�Ԥ�������ݺ���
def load_process_data():
    #��expolanet_data.csv�ļ�����ǰĿ¼���ж�ȡ����
    #���ڶ�ȡ����DataFrame���ͣ�����������df��д��ʾ��ȡ������
    #��pandas������ȡcsv�ļ��������дΪpd��
    df=pd.read_csv('exoplanet_data.csv',comment='#')

    #�Ա��������������x����ֵΪ����볤��a�ͺ���ָ��m
    x=df[['pl_orbsmax','st_mass']]

    #����������������y����ֵΪ�������
    y=df['pl_orbper']

    #ɾ��ȱʧֵ��
    #��Ϊ���Իع��㷨���ܴ���ȱʧֵ��NaN��null��������Ҫ�ֶ�����ȱʧֵ����ֹ��������ȱʧֵ���֡�
    
    data=pd.concat([x,y],axis=1)#pandas�����Ӻ�����axis=1��ʾˮƽ���ӣ��������ｫx��yˮƽ��������
    #�ٸ����ӣ����磺
    """
    ԭʼ���ݣ�
    x:
       pl_orbsmax  st_mass
    0    1.0        2.0
    1    2.0        NaN#ȱʧֵ
    2    3.0        3.0
    3    4.0        4.0

    y:
    0    10.0
    1    20.0
    2    NaN#ȱʧֵ
    3    40.0

    ���Ӻ�
       pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    1    2.0        NaN      20.0    # ��ȱʧֵ
    2    3.0        3.0      NaN     # ��ȱʧֵ
    3    4.0        4.0      40.0
    """

    data=data.dropna()#ɾ������ȱʧֵ����
    #ɾ��ȱʧֵ���������£�
    """
    pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    3    4.0        4.0      40.0
    """

    #���������Ǵ�ɾ�������������ȡ�Ա���x�������y
    x=data[['pl_orbsmax','st_mass']]
    y=data['pl_orbper']

    #��ô�����x��y�ͱ���ˣ�
    """
    x:
   pl_orbsmax  st_mass
    0    1.0        2.0
    3    4.0        4.0

    y:
    0    10.0
    3    40.0
    """

    #���ش������Ա���x�������y
    return x,y
```
</details>

<details>
<summary><strong>?? ����sklearnѵ��ģ��</strong></summary>

---

�õ������Ԥ�����ı����Ժ󣬾Ϳ���ѵ��ģ���ˡ���ô��������**sklearn**ѵ��ģ�͡�
�������� **train_test_split** ����������������ݺ�ѵ�����ݣ�  
```python
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�

    #test_size��ʾ�ж��ٵ������������ԣ�random_state=42��ʾ�������Ϊ42��������ӵ�������Ϊ��ÿ�����д���ʱ�����ֽ������ͬ��
    #����ѡ���������Ϊ42����Ϊ42��������ռ����أ�����
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)
```
���ţ�����LinearRegression����ѵ��ģ�ͣ�  
```python
    #������ѵ�����Իع�ģ��

    #��������scikit-learn���е�LinearRegression�����������Իع�ģ�ͣ�����model����LinearRegression���һ��ʵ����
    #LinearRegression����scikit-learn���е�һ�����Իع��࣬���ڹ���������y=ax_{1}+bx_{2}+c�ķ��̡�����y��������ҪԤ��Ĺ�����ڣ�x_{1}�������볤�ᣬx_{2}�������������
    model=LinearRegression()

    #����LinearRegression���е�fit����ѵ��ģ��
    model.fit(x_train,y_train)
```
Ȼ��ѵ������֮����õ�Ԥ��ֵ��mse��  
```python
    #��LinearRegression���е�predict��������Ԥ��
    y_pred=model.predict(x_test)

    #����mean_squared_error��������������
    mse=mean_squared_error(y_test,y_pred)
```
������Щ֮�����ǾͿ��Եõ�ѵ���Ͳ���ģ�͵ĺ���**train_test_model**�ˣ�  
```python
#ѵ���Ͳ���ģ�ͺ���
def train_test_model(x,y,test_size=0.7):
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�

    #test_size��ʾ�ж��ٵ������������ԣ�random_state=42��ʾ�������Ϊ42��������ӵ�������Ϊ��ÿ�����д���ʱ�����ֽ������ͬ��
    #����ѡ���������Ϊ42����Ϊ42��������ռ����أ�����
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)

    #������ѵ�����Իع�ģ��

    #��������scikit-learn���е�LinearRegression�����������Իع�ģ�ͣ�����model����LinearRegression���һ��ʵ����
    #LinearRegression����scikit-learn���е�һ�����Իع��࣬���ڹ���������y=ax_{1}+bx_{2}+c�ķ��̡�����y��������ҪԤ��Ĺ�����ڣ�x_{1}�������볤�ᣬx_{2}�������������
    model=LinearRegression()

    #����LinearRegression���е�fit����ѵ��ģ��
    model.fit(x_train,y_train)

    #��LinearRegression���е�predict��������Ԥ��
    y_pred=model.predict(x_test)

    #����mean_squared_error��������������
    mse=mean_squared_error(y_test,y_pred)

    #����model,mse,y_test,y_pred
    return model,mse,y_test,y_pred
```
</details>

<details>
<summary><strong>? �ֶ�ʹ����С���˷���������Իع�����</strong></summary>

---

������Ŀ��˵Ҫ**�ֶ�ʹ����С���˷���������Իع�����**���������ǻ���Ҫ�ó�����ʵ��**��С���˷�**��

- **��С���˷�**  
    ����**���Իع鷽��**Ϊ��  
    $$
    y=X*{\beta} + {\varepsilon}
    $$
    ����֪�����Ҫ**�������Իع�ģ���Ƿ���Ч**����������**�в�ƽ����**��������  
    $$
    l=\sum_{i=1}^n {\varepsilon_{i}}^2=(y-X*{\beta})^T(y-X*{\beta})=(y^T-{\beta}^T*X^T)(y-X*{\beta})=y^Ty-y^TX{\beta}-{\beta}^TX^Ty+{\beta}^TX^TX{\beta}  
    $$
    $$
    ������{\beta}^TX^Ty=(X{\beta})^Ty��(X{\beta})^T����������y��������������{\beta}^TX^Ty=({\beta}^TX^Ty)^T=y^TX{\beta}  
    $$
    $���������У�$
    $$
    l=y^Ty-2{\beta}^TX^Ty+{\beta}^TX^TX{\beta}
    $$
    ������Ҫʹ��**�в�ƽ����**��С�����Զ�������󵼣�Ȼ��õ�ʹ������С�Ľ⣺  
    $$
    \frac{\partial l}{\partial {\beta}}=-2X^Ty+2X^TX{\beta}
    $$
    $�������0��$
    $$
    -2X^Ty+2X^TX{\beta}=0
    $$
    $���Եõ���$
    $$
    {\beta}=(X^TX)^{-1}X^Ty
    $$
    ���������**���յĽ�����**��  
    ��ô���������Ǿ��ó���ʵ�־ͺ��ˡ�������**�ֶ�ʹ����С���˷�������Իع�����**�ĺ�����  
    ```python
    #�ֶ�ʵ����С���˷�ʵ�����Իع飨Build it from scratchӦ���������˼�ɣ�����
    def mannual_linear_regression(x,y,test_size=0.7):
        #�����ݼ�����Ϊѵ�����Ͳ��Լ�
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
        #���ƫ���Ҳ���ǽؾ�����Ѿ���x���������һ��ȫ�����1���Ա��ڳ��ֳ�����
        #���磺
        """
        ԭʼ���ݣ�
        x:
        |2 3|
        |4 5|
        ������
        x1=2,x2=3
        x1=4,x2=5
    
        ���ƫ�����
        x:
        |1 2 3|
        |1 4 5|
    
        �ͻ��ǣ�
        result:
        |1xc 2x��1 3x��2|
        |1xc 4x��1 5x��2|
        """
        #���ƫ����
        x_train_b=np.c_[np.ones(x_train.shape[0]),x_train]
    
        #������С���˷��Ľ����⣺��=(X^T*X)^-1*X^T*y
        results=np.linalg.inv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)
    
        #������Լ���Ԥ��ֵ
        x_test_b=np.c_[np.ones(x_test.shape[0]),x_test]
        y_pred=x_test_b.dot(results)
    
        #����������
        mse=mean_squared_error(y_test,y_pred)
    
        #���ؽ��
        #results[0]�ǽؾ��results[1:]��ϵ����
        return results[0],results[1:],y_test,y_pred,mse
    ```
</details>

<details>
<summary><strong>? ���ӻ����</strong></summary>

---

�����������ݣ�ѵ����ģ���Ժ���������**matplotlib��**�����ӻ������  
��ôΪ�˸��õĿ��ӻ�������������һ��**������slider**������test_size��������ͬ�Ĳ�������ռ�Ȼᵼ�½����ô�仯������Ҫע����Ǹ���ͼ��������Ҫ**slider����һ��update����**��
�����ڴ���ע�����Ѿ�����ϸ�ˣ���������ֱ�Ӹ���**visualize_results**���������룺  
```python
#����plt���ӻ�Ԥ����
def visualize_results(x,y,mode=1):
    #����ͼ�δ��ڣ�������С����Ϊ10x6Ӣ��
    fig=plt.figure(figsize=(10,6))

    #����GridSpec�������ڿ�����ͼ�Ĳ���
    #2,1��ʾ����һ��2��1�е���ͼ
    #height_ratios=[1,4]��ʾ��һ�еĸ߶�Ϊ1���ڶ��еĸ߶�Ϊ4
    #hspace=0.2��ʾ��ͼ֮��Ĵ�ֱ���Ϊ0.2
    gs=fig.add_gridspec(2,1,height_ratios=[1,4],hspace=0.2)

    #������ĵ�һ�д���һ����ͼ��������ʾ���Իع鷽�̣�����ָ��
    #gs[0]��ʾ����ĵ�һ��
    ax_text=fig.add_subplot(gs[0])

    #������ĵڶ��д���һ��ɢ��ͼ��ͼ
    #gs[1]��ʾ����ĵڶ���
    ax_scatter=fig.add_subplot(gs[1])

    #������ͼ���֣��ڵײ�����0.2�Ŀհ�������û�����
    plt.subplots_adjust(bottom=0.2)
    
    if mode==1:
        #��ȡtrain_test_model�ķ���ֵ
        model,mse,y_test,y_pred=train_test_model(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
    elif mode==2:
        #��ȡmannual_linear_regression�ķ���ֵ
        intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'

    #��ȡMSE�ı�
    mse_text=f'������$MSE$����{mse:.3f}'

    #plt.scatter()������ɢ��ͼ�����е�һ��������x�ᣨ��ʾ��ʵֵ�����ڶ���������y�ᣨ��ʾԤ��ֵ����������������͸����
    scatter=ax_scatter.scatter(y_test,y_pred,alpha=0.5,label='pred vs. true')

    #��ȡ���������Сֵ�����ֵ
    ax_min=min(y_test.min(),y_pred.min())
    ax_max=max(y_test.max(),y_pred.max())

    #���������Ԥ���ߣ�ideal����Ҳ���ǶԽ���
    #[y_test.min(),y_test.max()]��x��������յ�
    #[y_test.min(),y_test.max()]��y��������յ�
    #'r--'����ɫ���ߣ�r����red��ɫ��--�������ߣ�
    #lw=2���߿�Ϊ2
    line,=ax_scatter.plot([ax_min,ax_max],[ax_min,ax_max],'r--',lw=2,label='ideal')

    #����ɢ��ͼ�������᷶Χ��ʹ֮���һ��������
    ax_scatter.set_xlim(ax_min,ax_max)
    ax_scatter.set_ylim(ax_min,ax_max)
    #����ɢ��ͼ�����������
    ax_scatter.set_aspect('equal',adjustable='box')
        
    #����һ�������������ڿ��Ʋ��Լ��ı���test_size
    #0.2����������x��
    #0.1����������y��
    #0.6���������Ŀ��
    #0.03���������ĸ߶�
    ax_slider=plt.axes([0.2,0.1,0.6,0.03])
    #����һ�����ڷ�ΧΪ0-1�Ļ����������ҳ�ʼֵΪ0.7
    slider=Slider(ax_slider,'Test Size',0,1,valinit=0.7)

    #�ı���Ϣ�ر�������
    ax_text.axis('off')    

    #��ӵ�ax_text��
    #0.05�����ı������Ͻǵ�x����
    #0.5�����ı������Ͻǵ�y����
    #transformָ������ϵ��transAxes(�������ϵ)�����귶ΧΪ[0,1]
    #verticalalignment='center'��ʾ�ı���ֱ������(0.05,0.5)
    #bbox=dict()�����ı���ı����ͱ߿���ʽ��boxstyle='round'�ı���ΪԲ�Ǿ��Σ�facecolor='white'������ɫΪ��ɫ,alpha=0.8͸����Ϊ0.8
    ax_text.text(0.05,0.5,f'���Իع鷽�̣�\n{equ}\n\n����ָ�꣺\n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    #���û������е�ֵ������test_size
    def update(val):
        #��ȡ�������е�ֵ�����и�ֵ
        test_size=slider.val
        if mode==1:
            #��������test_size��ȡ��������
            model,mse,y_test,y_pred=train_test_model(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
        elif mode==2:
            #��������test_size��ȡ��������
            intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'
            
        #��ȡMSE�ı�
        mse_text=f'������$MSE$����{mse:.3f}'

        #����ɢ��ͼ
        scatter.set_offsets(np.c_[y_test,y_pred])

        #�����ı�ע��
        #������ԭ��������
        ax_text.clear()
        #�ر�������
        ax_text.axis('off')
        #����µ��ı�ע��
        ax_text.text(0.05,0.5,f'���Իع鷽�̣� \n{equ}\n\n����ָ�꣺ \n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        #��������������ֵ����Сֵ
        ax_min=min(y_test.min(),y_pred.min())
        ax_max=max(y_test.max(),y_pred.max())

        #��������Ԥ����
        line.set_data([ax_min,ax_max],[ax_min,ax_max])

        #����������ķ�Χ
        ax_scatter.set_xlim(ax_min,ax_max)
        ax_scatter.set_ylim(ax_min,ax_max)
        #����������ı���
        ax_scatter.set_aspect('equal',adjustable='box')

        #����ͼ����������draw_idle()����draw()����Ϊdraw_idle()���ʺϽ���ʽ�Ķ������绬����
        fig.canvas.draw_idle()

    #���û�������on_changed����������ͼ��
    slider.on_changed(update)

    #�����Ͻ����ͼ��
    ax_scatter.legend(loc='upper left')

    #����x���ǩ��ʾ��ͼ�и�����T_true��
    ax_scatter.set_xlabel('T_true')

    #����y���ǩ��ʾ��ͼ�и�����T_pred��
    ax_scatter.set_ylabel('T_pred')

    #����ͼ��ı��⣨ʾ��ͼ�и�����true vs. pred T��
    ax_scatter.set_title('true vs. pred T')

    ax_scatter.grid(True)

    #��ʾͼ��
    plt.show()
```
</details>

### ��������
<details>
<summary><strong>? ��������</strong></summary>

---

```python
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

# ����matplotlib֧��������ʾ
plt.rcParams['font.sans-serif']=['SimHei']#����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus']=False#����������ʾ����
plt.rcParams['mathtext.fontset'] ='cm'#���� LaTeX ����

#���غ�Ԥ�������ݺ���
def load_process_data():
    #��expolanet_data.csv�ļ�����ǰĿ¼���ж�ȡ����
    #���ڶ�ȡ����DataFrame���ͣ�����������df��д��ʾ��ȡ������
    #��pandas������ȡcsv�ļ��������дΪpd��
    df=pd.read_csv('exoplanet_data.csv',comment='#')

    #�Ա��������������x����ֵΪ����볤��a�ͺ���ָ��m
    x=df[['pl_orbsmax','st_mass']]

    #����������������y����ֵΪ�������
    y=df['pl_orbper']

    #ɾ��ȱʧֵ��
    #��Ϊ���Իع��㷨���ܴ���ȱʧֵ��NaN��null��������Ҫ�ֶ�����ȱʧֵ����ֹ��������ȱʧֵ���֡�
    
    data=pd.concat([x,y],axis=1)#pandas�����Ӻ�����axis=1��ʾˮƽ���ӣ��������ｫx��yˮƽ��������
    #�ٸ����ӣ����磺
    """
    ԭʼ���ݣ�
    x:
       pl_orbsmax  st_mass
    0    1.0        2.0
    1    2.0        NaN#ȱʧֵ
    2    3.0        3.0
    3    4.0        4.0

    y:
    0    10.0
    1    20.0
    2    NaN#ȱʧֵ
    3    40.0

    ���Ӻ�
       pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    1    2.0        NaN      20.0    # ��ȱʧֵ
    2    3.0        3.0      NaN     # ��ȱʧֵ
    3    4.0        4.0      40.0
    """

    data=data.dropna()#ɾ������ȱʧֵ����
    #ɾ��ȱʧֵ���������£�
    """
    pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    3    4.0        4.0      40.0
    """

    #���������Ǵ�ɾ�������������ȡ�Ա���x�������y
    x=data[['pl_orbsmax','st_mass']]
    y=data['pl_orbper']

    #��ô�����x��y�ͱ���ˣ�
    """
    x:
   pl_orbsmax  st_mass
    0    1.0        2.0
    3    4.0        4.0

    y:
    0    10.0
    3    40.0
    """

    #���ش������Ա���x�������y
    return x,y

#ѵ���Ͳ���ģ�ͺ���
def train_test_model(x,y,test_size=0.7):
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�

    #test_size��ʾ�ж��ٵ������������ԣ�random_state=42��ʾ�������Ϊ42��������ӵ�������Ϊ��ÿ�����д���ʱ�����ֽ������ͬ��
    #����ѡ���������Ϊ42����Ϊ42��������ռ����أ�����
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)

    #������ѵ�����Իع�ģ��

    #��������scikit-learn���е�LinearRegression�����������Իع�ģ�ͣ�����model����LinearRegression���һ��ʵ����
    #LinearRegression����scikit-learn���е�һ�����Իع��࣬���ڹ���������y=ax_{1}+bx_{2}+c�ķ��̡�����y��������ҪԤ��Ĺ�����ڣ�x_{1}�������볤�ᣬx_{2}�������������
    model=LinearRegression()

    #����LinearRegression���е�fit����ѵ��ģ��
    model.fit(x_train,y_train)

    #��LinearRegression���е�predict��������Ԥ��
    y_pred=model.predict(x_test)

    #����mean_squared_error��������������
    mse=mean_squared_error(y_test,y_pred)

    #����model,mse,y_test,y_pred
    return model,mse,y_test,y_pred

#�ֶ�ʵ����С���˷�ʵ�����Իع飨Build it from scratchӦ���������˼�ɣ�����
def mannual_linear_regression(x,y,test_size=0.7):
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    #���ƫ���Ҳ���ǽؾ�����Ѿ���x���������һ��ȫ�����1���Ա��ڳ��ֳ�����
    #���磺
    """
    ԭʼ���ݣ�
    x:
    |2 3|
    |4 5|
    ������
    x1=2,x2=3
    x1=4,x2=5

    ���ƫ�����
    x:
    |1 2 3|
    |1 4 5|

    �ͻ��ǣ�
    result:
    |1xc 2x��1 3x��2|
    |1xc 4x��1 5x��2|
    """
    #���ƫ����
    x_train_b=np.c_[np.ones(x_train.shape[0]),x_train]

    #������С���˷��Ľ����⣺��=(X^T*X)^-1*X^T*y
    results=np.linalg.inv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)

    #������Լ���Ԥ��ֵ
    x_test_b=np.c_[np.ones(x_test.shape[0]),x_test]
    y_pred=x_test_b.dot(results)

    #����������
    mse=mean_squared_error(y_test,y_pred)

    #���ؽ��
    #results[0]�ǽؾ��results[1:]��ϵ����
    return results[0],results[1:],y_test,y_pred,mse

#����plt���ӻ�Ԥ����
def visualize_results(x,y,mode=1):
    #����ͼ�δ��ڣ�������С����Ϊ10x6Ӣ��
    fig=plt.figure(figsize=(10,6))

    #����GridSpec�������ڿ�����ͼ�Ĳ���
    #2,1��ʾ����һ��2��1�е���ͼ
    #height_ratios=[1,4]��ʾ��һ�еĸ߶�Ϊ1���ڶ��еĸ߶�Ϊ4
    #hspace=0.2��ʾ��ͼ֮��Ĵ�ֱ���Ϊ0.2
    gs=fig.add_gridspec(2,1,height_ratios=[1,4],hspace=0.2)

    #������ĵ�һ�д���һ����ͼ��������ʾ���Իع鷽�̣�����ָ��
    #gs[0]��ʾ����ĵ�һ��
    ax_text=fig.add_subplot(gs[0])

    #������ĵڶ��д���һ��ɢ��ͼ��ͼ
    #gs[1]��ʾ����ĵڶ���
    ax_scatter=fig.add_subplot(gs[1])

    #������ͼ���֣��ڵײ�����0.2�Ŀհ�������û�����
    plt.subplots_adjust(bottom=0.2)
    
    if mode==1:
        #��ȡtrain_test_model�ķ���ֵ
        model,mse,y_test,y_pred=train_test_model(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
    elif mode==2:
        #��ȡmannual_linear_regression�ķ���ֵ
        intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'

    #��ȡMSE�ı�
    mse_text=f'������$MSE$����{mse:.3f}'

    #plt.scatter()������ɢ��ͼ�����е�һ��������x�ᣨ��ʾ��ʵֵ�����ڶ���������y�ᣨ��ʾԤ��ֵ����������������͸����
    scatter=ax_scatter.scatter(y_test,y_pred,alpha=0.5,label='pred vs. true')

    #��ȡ���������Сֵ�����ֵ
    ax_min=min(y_test.min(),y_pred.min())
    ax_max=max(y_test.max(),y_pred.max())

    #���������Ԥ���ߣ�ideal����Ҳ���ǶԽ���
    #[y_test.min(),y_test.max()]��x��������յ�
    #[y_test.min(),y_test.max()]��y��������յ�
    #'r--'����ɫ���ߣ�r����red��ɫ��--�������ߣ�
    #lw=2���߿�Ϊ2
    line,=ax_scatter.plot([ax_min,ax_max],[ax_min,ax_max],'r--',lw=2,label='ideal')

    #����ɢ��ͼ�������᷶Χ��ʹ֮���һ��������
    ax_scatter.set_xlim(ax_min,ax_max)
    ax_scatter.set_ylim(ax_min,ax_max)
    #����ɢ��ͼ�����������
    ax_scatter.set_aspect('equal',adjustable='box')
        
    #����һ�������������ڿ��Ʋ��Լ��ı���test_size
    #0.2����������x��
    #0.1����������y��
    #0.6���������Ŀ��
    #0.03���������ĸ߶�
    ax_slider=plt.axes([0.2,0.1,0.6,0.03])
    #����һ�����ڷ�ΧΪ0-1�Ļ����������ҳ�ʼֵΪ0.7
    slider=Slider(ax_slider,'Test Size',0,1,valinit=0.7)

    #�ı���Ϣ�ر�������
    ax_text.axis('off')    

    #��ӵ�ax_text��
    #0.05�����ı������Ͻǵ�x����
    #0.5�����ı������Ͻǵ�y����
    #transformָ������ϵ��transAxes(�������ϵ)�����귶ΧΪ[0,1]
    #verticalalignment='center'��ʾ�ı���ֱ������(0.05,0.5)
    #bbox=dict()�����ı���ı����ͱ߿���ʽ��boxstyle='round'�ı���ΪԲ�Ǿ��Σ�facecolor='white'������ɫΪ��ɫ,alpha=0.8͸����Ϊ0.8
    ax_text.text(0.05,0.5,f'���Իع鷽�̣�\n{equ}\n\n����ָ�꣺\n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    #���û������е�ֵ������test_size
    def update(val):
        #��ȡ�������е�ֵ�����и�ֵ
        test_size=slider.val
        if mode==1:
            #��������test_size��ȡ��������
            model,mse,y_test,y_pred=train_test_model(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
        elif mode==2:
            #��������test_size��ȡ��������
            intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'
            
        #��ȡMSE�ı�
        mse_text=f'������$MSE$����{mse:.3f}'

        #����ɢ��ͼ
        scatter.set_offsets(np.c_[y_test,y_pred])

        #�����ı�ע��
        #������ԭ��������
        ax_text.clear()
        #�ر�������
        ax_text.axis('off')
        #����µ��ı�ע��
        ax_text.text(0.05,0.5,f'���Իع鷽�̣� \n{equ}\n\n����ָ�꣺ \n{mse_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        #��������������ֵ����Сֵ
        ax_min=min(y_test.min(),y_pred.min())
        ax_max=max(y_test.max(),y_pred.max())

        #��������Ԥ����
        line.set_data([ax_min,ax_max],[ax_min,ax_max])

        #����������ķ�Χ
        ax_scatter.set_xlim(ax_min,ax_max)
        ax_scatter.set_ylim(ax_min,ax_max)
        #����������ı���
        ax_scatter.set_aspect('equal',adjustable='box')

        #����ͼ����������draw_idle()����draw()����Ϊdraw_idle()���ʺϽ���ʽ�Ķ������绬����
        fig.canvas.draw_idle()

    #���û�������on_changed����������ͼ��
    slider.on_changed(update)

    #�����Ͻ����ͼ��
    ax_scatter.legend(loc='upper left')

    #����x���ǩ��ʾ��ͼ�и�����T_true��
    ax_scatter.set_xlabel('T_true')

    #����y���ǩ��ʾ��ͼ�и�����T_pred��
    ax_scatter.set_ylabel('T_pred')

    #����ͼ��ı��⣨ʾ��ͼ�и�����true vs. pred T��
    ax_scatter.set_title('true vs. pred T')

    ax_scatter.grid(True)

    #��ʾͼ��
    plt.show()

#������
def main(mode=1):
    #���غ�Ԥ��������
    x,y=load_process_data()

    #���ӻ�Ԥ����
    visualize_results(x,y,mode)

#�����ǰ�ļ��������򣬶����Ǳ�����ģ�飬��ִ��main����
if __name__=='__main__':
    if len(sys.argv)>=2:
        main(int(sys.argv[1]))


```
</details>

<details>
<summary><strong>? ��չ����</strong></summary>

---

```python
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

#����matplotlib֧��������ʾ
plt.rcParams['font.sans-serif']=['SimHei'] #����������ʾ���ı�ǩ
plt.rcParams['axes.unicode_minus']=False #����������ʾ����
plt.rcParams['mathtext.fontset']='cm'#���� LaTeX ����

#���غ�Ԥ�������ݺ���
def load_process_data():
    #��expolanet_data.csv�ļ�����ǰĿ¼���ж�ȡ����
    #���ڶ�ȡ����DataFrame���ͣ�����������df��д��ʾ��ȡ������
    #��pandas������ȡcsv�ļ��������дΪpd��
    df=pd.read_csv('exoplanet_data.csv',comment='#')

    #�Ա��������������x����ֵΪ����볤��a�ͺ�������m
    x=df[['pl_orbsmax','st_mass']]

    #����������������y����ֵΪ�������
    y=df['pl_orbper']

    #ɾ��ȱʧֵ��
    #��Ϊ���Իع��㷨���ܴ���ȱʧֵ��NaN��null��������Ҫ�ֶ�����ȱʧֵ����ֹ��������ȱʧֵ���֡�
    
    data=pd.concat([x,y],axis=1) #pandas�����Ӻ�����axis=1��ʾˮƽ���ӣ��������ｫx��yˮƽ��������
    #�ٸ����ӣ����磺
    """
    ԭʼ���ݣ�
    x:
       pl_orbsmax  st_mass
    0    1.0        2.0
    1    2.0        NaN # ȱʧֵ
    2    3.0        3.0
    3    4.0        4.0

    y:
    0    10.0
    1    20.0
    2    NaN # ȱʧֵ
    3    40.0

    ���Ӻ�
       pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    1    2.0        NaN      20.0    # ��ȱʧֵ
    2    3.0        3.0      NaN     # ��ȱʧֵ
    3    4.0        4.0      40.0
    """

    data=data.dropna()#ɾ������ȱʧֵ����
    #ɾ��ȱʧֵ���������£�
    """
    pl_orbsmax  st_mass  pl_orbper
    0    1.0        2.0      10.0
    3    4.0        4.0      40.0
    """

    #���������Ǵ�ɾ�������������ȡ�Ա���x�������y
    x=data[['pl_orbsmax','st_mass']]
    y=data['pl_orbper']

    #��ô�����x��y�ͱ���ˣ�
    """
    x:
   pl_orbsmax  st_mass
    0    1.0        2.0
    3    4.0        4.0

    y:
    0    10.0
    3    40.0
    """

    #extend.py��չ�����¼ӣ������任���ҿ��Լ�С����֮��ļ�϶
    x=np.log(abs(x))#�Ӿ���ֵ���Ա��⸺ֵ
    y=np.log(abs(y))

    #���ش������Ա���x�������y
    return x,y

#ѵ���Ͳ���ģ�ͺ���
def train_test_model(x,y,test_size=0.7):
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�

    #test_size��ʾ�ж��ٵ������������ԣ�random_state=42��ʾ�������Ϊ42��������ӵ�������Ϊ��ÿ�����д���ʱ�����ֽ������ͬ��
    #����ѡ���������Ϊ42����Ϊ42��������ռ����أ�����
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)

    #������ѵ�����Իع�ģ��

    #��������scikit-learn���е�LinearRegression�����������Իع�ģ�ͣ�����model����LinearRegression���һ��ʵ����
    #LinearRegression����scikit-learn���е�һ�����Իع��࣬���ڹ���������y=ax_{1}+bx_{2}+c�ķ��̡�����y��������ҪԤ��Ĺ�����ڣ�x_{1}�������볤�ᣬx_{2}�������������
    model=LinearRegression()

    #����LinearRegression���е�fit����ѵ��ģ��
    model.fit(x_train,y_train)

    #��LinearRegression���е�predict��������Ԥ��
    y_pred=model.predict(x_test)

    #extend.py��չ�����¼ӣ�����R?����
    r2=model.score(x_test,y_test)

    #����mean_squared_error��������������
    mse=mean_squared_error(y_test,y_pred)

    #����model,mse,r2,y_test,y_pred
    return model,mse,r2,y_test,y_pred

#�ֶ�ʵ����С���˷�ʵ�����Իع飨Build it from scratchӦ���������˼�ɣ�����
def mannual_linear_regression(x,y,test_size=0.7):
    #�����ݼ�����Ϊѵ�����Ͳ��Լ�
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    #���ƫ���Ҳ���ǽؾ�����Ѿ���x���������һ��ȫ�����1���Ա��ڳ��ֳ�����
    #���磺
    """
    ԭʼ���ݣ�
    x:
    |2 3|
    |4 5|
    ������
    x1=2,x2=3
    x1=4,x2=5

    ���ƫ�����
    x:
    |1 2 3|
    |1 4 5|

    �ͻ��ǣ�
    result:
    |1xc 2x��1 3x��2|
    |1xc 4x��1 5x��2|
    """
    #���ƫ����
    x_train_b=np.c_[np.ones(x_train.shape[0]),x_train]

    #������С���˷��Ľ����⣺��=(X^T*X)^-1*X^T*y
    results=np.linalg.inv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)

    #������Լ���Ԥ��ֵ
    x_test_b=np.c_[np.ones(x_test.shape[0]),x_test]
    y_pred=x_test_b.dot(results)

    #����������
    mse=mean_squared_error(y_test,y_pred)

    #extend.py��չ�����¼ӣ�����R?����
    #R?=1-(�в�ƽ����/��ƽ����)
    #�в�ƽ����=sum((y_test-y_pred)**2)
    l=np.sum((y_test-y_pred)**2)
    #��ƽ����=sum((y_test-y_test.mean())**2)
    r=np.sum((y_test-y_test.mean())**2)
    r2=1-(l/r)

    #���ؽ��
    #results[0]�ǽؾ��results[1:]��ϵ����
    return results[0],results[1:],y_test,y_pred,mse,r2

#����plt���ӻ�Ԥ����
def visualize_results(x,y,mode=1):
    #����ͼ�δ��ڣ�������С����Ϊ10x6Ӣ��
    fig=plt.figure(figsize=(10,6))

    #����GridSpec�������ڿ�����ͼ�Ĳ���
    #2,1��ʾ����һ��2��1�е���ͼ
    #height_ratios=[1,4]��ʾ��һ�еĸ߶�Ϊ1���ڶ��еĸ߶�Ϊ4
    #hspace=0.2��ʾ��ͼ֮��Ĵ�ֱ���Ϊ0.2
    gs=fig.add_gridspec(2,1,height_ratios=[1,4],hspace=0.2)

    #������ĵ�һ�д���һ����ͼ��������ʾ���Իع鷽�̣�����ָ��
    #gs[0]��ʾ����ĵ�һ��
    ax_text=fig.add_subplot(gs[0])

    #������ĵڶ��д���һ��ɢ��ͼ��ͼ
    #gs[1]��ʾ����ĵڶ���
    ax_scatter=fig.add_subplot(gs[1])

    #������ͼ���֣��ڵײ�����0.2�Ŀհ�������û�����
    plt.subplots_adjust(bottom=0.2)
    
    if mode==1:
        #��ȡtrain_test_model�ķ���ֵ
        model,mse,r2,y_test,y_pred=train_test_model(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
    elif mode==2:
        #��ȡmannual_linear_regression�ķ���ֵ
        intercept,coef,y_test,y_pred,mse,r2=mannual_linear_regression(x,y,test_size=0.7)

        #��ȡ���Իع鷽�̵��ı�
        equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'

    #��ȡMSE�ı�
    mse_text=f'������$MSE$����{mse:.3f}'

    #��ȡR?�ı�
    r2_text=f'����ϵ����$R^2$����{r2:.3f}'

    #��¼����ı�
    error_text=f'{mse_text}\n{r2_text}'

    #plt.scatter()������ɢ��ͼ�����е�һ��������x�ᣨ��ʾ��ʵֵ�����ڶ���������y�ᣨ��ʾԤ��ֵ����������������͸����
    scatter=ax_scatter.scatter(y_test,y_pred,alpha=0.5,label='pred vs. true')

    #��ȡ���������Сֵ�����ֵ
    ax_min=min(y_test.min(),y_pred.min())
    ax_max=max(y_test.max(),y_pred.max())

    #���������Ԥ���ߣ�ideal����Ҳ���ǶԽ���
    #[y_test.min(),y_test.max()]��x��������յ�
    #[y_test.min(),y_test.max()]��y��������յ�
    #'r--'����ɫ���ߣ�r����red��ɫ��--�������ߣ�
    #lw=2���߿�Ϊ2
    line,=ax_scatter.plot([ax_min,ax_max],[ax_min,ax_max],'r--',lw=2,label='ideal')

    #����ɢ��ͼ�������᷶Χ��ʹ֮���һ��������
    ax_scatter.set_xlim(ax_min,ax_max)
    ax_scatter.set_ylim(ax_min,ax_max)
    #����ɢ��ͼ�����������
    ax_scatter.set_aspect('equal',adjustable='box')
        
    #����һ�������������ڿ��Ʋ��Լ��ı���test_size
    #0.2����������x��
    #0.1����������y��
    #0.6���������Ŀ��
    #0.03���������ĸ߶�
    ax_slider=plt.axes([0.2,0.1,0.6,0.03])
    #����һ�����ڷ�ΧΪ0-1�Ļ����������ҳ�ʼֵΪ0.7
    slider=Slider(ax_slider,'Test Size',0,1,valinit=0.7)

    #�ı���Ϣ�ر�������
    ax_text.axis('off')    

    #��ӵ�ax_text��
    #0.05�����ı������Ͻǵ�x����
    #0.5�����ı������Ͻǵ�y����
    #transformָ������ϵ��transAxes(�������ϵ)�����귶ΧΪ[0,1]
    #verticalalignment='center'��ʾ�ı���ֱ������(0.05,0.5)
    #bbox=dict()�����ı���ı����ͱ߿���ʽ��boxstyle='round'�ı���ΪԲ�Ǿ��Σ�facecolor='white'������ɫΪ��ɫ,alpha=0.8͸����Ϊ0.8
    ax_text.text(0.05,0.5,f'���Իع鷽�̣�\n{equ}\n\n����ָ�꣺\n{error_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    #���û������е�ֵ������test_size
    def update(val):
        #��ȡ�������е�ֵ�����и�ֵ
        test_size=slider.val
        if mode==1:
            #��������test_size��ȡ��������
            model,mse,r2,y_test,y_pred=train_test_model(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {model.coef_[0]:.3f} * pl_orbsmax + {model.coef_[1]:.3f} * st_mass + {model.intercept_:.3f}'
        elif mode==2:
            #��������test_size��ȡ��������
            intercept,coef,y_test,y_pred,mse=mannual_linear_regression(x,y,test_size=test_size)
            #��ȡ���Իع鷽�̵��ı�
            equ=f'pl_order = {coef[0]:.3f} * pl_orbsmax + {coef[1]:.3f} * st_mass + {intercept:.3f}'
            
        #��ȡMSE�ı�
        mse_text=f'������$MSE$����{mse:.3f}'

        #��ȡR?�ı�
        r2_text=f'����ϵ����$R^2$����{r2:.3f}'

        #��¼����ı�
        error_text=f'{mse_text}\n{r2_text}'

        #����ɢ��ͼ
        scatter.set_offsets(np.c_[y_test,y_pred])

        #�����ı�ע��
        #������ԭ��������
        ax_text.clear()
        #�ر�������
        ax_text.axis('off')
        #����µ��ı�ע��
        ax_text.text(0.05,0.5,f'���Իع鷽�̣� \n{equ}\n\n����ָ�꣺ \n{error_text}',transform=ax_text.transAxes,verticalalignment='center',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        #��������������ֵ����Сֵ
        ax_min=min(y_test.min(),y_pred.min())
        ax_max=max(y_test.max(),y_pred.max())

        #��������Ԥ����
        line.set_data([ax_min,ax_max],[ax_min,ax_max])

        #����������ķ�Χ
        ax_scatter.set_xlim(ax_min,ax_max)
        ax_scatter.set_ylim(ax_min,ax_max)
        #����������ı���
        ax_scatter.set_aspect('equal',adjustable='box')

        #����ͼ����������draw_idle()����draw()����Ϊdraw_idle()���ʺϽ���ʽ�Ķ������绬����
        fig.canvas.draw_idle()

    #���û�������on_changed����������ͼ��
    slider.on_changed(update)

    #�����Ͻ����ͼ��
    ax_scatter.legend(loc='upper left')

    #����x���ǩ��ʾ��ͼ�и�����T_true��
    ax_scatter.set_xlabel('T_true')

    #����y���ǩ��ʾ��ͼ�и�����T_pred��
    ax_scatter.set_ylabel('T_pred')

    #����ͼ��ı��⣨ʾ��ͼ�и�����true vs. pred T��
    ax_scatter.set_title('true vs. pred T')

    ax_scatter.grid(True)

    #��ʾͼ��
    plt.show()

#������
def main(mode=1):
    #���غ�Ԥ��������
    x,y=load_process_data()

    #���ӻ�Ԥ����
    visualize_results(x,y,mode)

#�����ǰ�ļ��������򣬶����Ǳ�����ģ�飬��ִ��main����
if __name__=='__main__':
    if len(sys.argv)>=2:
        main(int(sys.argv[1]))
```
</details>

### ���н�ͼ

<details>
<summary><strong>? ��������</strong></summary>

---

- **base_sklearn.py**
    ![image1.png](image1.png)
  
- **base_mannual.py**
    ![image2.png](image2.png)
  
</details>

<details>
<summary><strong>? ��չ����</strong></summary>

---
  
- **extend_sklearn.py**
    ![image3.png](image3.png)
  
- **extend_mannual.py**
    ![image4.png](image4.png)
  
</details>

### �ܽ�

<details>
<summary><strong>? �ܽ�</strong></summary>

---

ȷʵ�ǹ��ڻ���ѧϰ�����Իع�ģ�͵ĺ��⣬����Ҳѧ���˺ܶ��µĿ⣬����scikit-learn��pandas��matplotlib��numpy�ȡ�������ǰҲֻ��ͣ���������ϣ������ͨ������ʵ����һ�£�ӡ�������
</details>
