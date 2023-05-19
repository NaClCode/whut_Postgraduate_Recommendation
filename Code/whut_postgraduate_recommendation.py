import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
import prophet
import pdfplumber
import wordcloud

class preliminary_analysis:
    def __init__(self, data):
        self.data = data

    def show_size(self):
        [print(self.data[i].shape) for i in range(0, 6)] #展示保研人数数据集大小

    def show_col(self):
        [print(self.data[i].columns) for i in range(0, 6)] #展示保研特征类型

    def show_head(self):
        [print(self.data[i].head()) for i in range(0, 6)] #展示前五个值

    def showna(self):
        [print(self.data[i].isna().sum()) for i in range(0, 6)] #展示缺失值
        
    def fillna(self):
        [self.data[i].fillna('', inplace=True) for i in range(0, 6)]#用''填充缺失值
    
class postgraduate_recommendation_type:
    def __init__(self, data):
        self.data = data

    def pie(self, values, index, explode, title):
        #此代码大体参照matplotlib的一个example，使饼图变得更加美观
        wedges, texts = plt.pie(values, wedgeprops=dict(width=0.5), startangle=-40)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            plt.annotate(index[i], xy=(x, y), xytext=(1.3*np.sign(x) * explode[i], 1.2 * y * explode[i]),
                        horizontalalignment=horizontalalignment, **kw)
        plt.title(title)

    def pie_img(self):
        plt.style.use('_mpl-gallery-nogrid')
        plt.rc('font', family = 'SimHei', size = 9)
        plt.rc('axes', unicode_minus = False)
        explode = [(1, 1, 1.5, 1), (1, 1, 1.5, 1), (1, 1), (1, 1, 1), (1, 1, 1.5, 1), (1, 1, 1.5, 1)] #解决饼图文字重叠
        plt.figure(figsize=(9, 10))
        for i in range(0, 6):
            plt.subplot(3, 2, i + 1)
            ret = self.data[i]['类别'].value_counts()
            self.pie(ret.values, ret.index, explode[i], f'20{17 + i}年保研类型')

    def line_img(self):
        plt.figure(figsize=(6, 5))
        data_A = [];data_B = []
        for i in range(0, 6):
            data_A.append(self.data[i][self.data[i]['类别'] == 'A'].shape[0])
            data_B.append(self.data[i][self.data[i]['类别'] == 'B'].shape[0])
        data_A_B = pd.DataFrame({'A':data_A, 'B':data_B}, index=range(2017, 2023))
        plt.title('保研类型人数趋势图')
        sns.lineplot(data_A_B, markers = 'o')

class postgraduate_recommendation_college:
    def __init__(self, data):
        self.data = data
        
    #画各学院保研人数占比与A保同学占比
    def bar_img(self, type_str):
        sns.set_theme(style="whitegrid")
        plt.rc('font', family = 'SimHei', size = 9)
        plt.rc('axes', unicode_minus = False)
        plt.figure(figsize=(15, 12))
        for i in range(0, 6):
            plt.subplot(3, 2, i + 1)
            plt.title(f'20{17 + i}年保研人数')
            tmp = self.data[i][type_str].value_counts().head(25)
            tmp_ = self.data[i][self.data[i]['类别'] == 'A'][type_str].value_counts().head(25)
            sns.set_color_codes("pastel")
            sns.barplot(x = tmp.values, y = tmp.index, color='b', label = '总人数')
            sns.set_color_codes("muted")
            sns.barplot(x = tmp_.values, y = tmp_.index, color='b', label = 'A保人数')
            plt.legend(ncol=2, loc="lower right", frameon=True)
    

class postgraduate_recommendation_preson:
    def __init__(self, data):
        self.data = data
        self.data_person, self.sum_person = self.get_person_data()

    #得到以往保研人数趋势
    def get_person_data(self):
        values = []
        index = ['理学院', '信息学院', '机电学院', '管理学院', '材料学院', '自动化学院', '计算机智能学院']
        columns = range(2017, 2023, 1)
        for i in index:
            tmp = []
            for j in range(0, 6):
                #解决计算机学院名称的变化问题
                if j < 5 and i == '计算机智能学院': i = '计算机学院'
                elif j == 5 and i == '计算机学院': i = '计算机智能学院'
                tmp.append(self.data[j]['学院名称'].value_counts()[i])
            values.append(tmp)

        data_person = pd.DataFrame(values, index = index, columns= columns).T
        sum_person = []
        for i in range(0, 6):
            sum_person.append(self.data[i].shape[0])
        sum_person = pd.Series(sum_person, index=range(2017, 2023))
        return data_person, sum_person

    #画折线图
    def line(self):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title('部分学院保研人数')
        sns.lineplot(self.data_person)
        plt.legend(loc = 'lower right')
        plt.subplot(2, 1, 2)
        plt.title('全校保研人数')
        sns.lineplot(self.sum_person)

class postgraduate_recommendation_preson_predict(postgraduate_recommendation_preson):
    def __init__(self, data):
        super().__init__(data)
        self.model = prophet.Prophet()
        self.model.fit(pd.DataFrame({'ds':range(2017, 2023), 'y':self.sum_person.values}))
    def RMSE(self):
        forecast = self.model.predict(pd.DataFrame({'ds':range(2017, 2023)}))
        train_len = len(self.sum_person.values)
        rmse = np.sqrt(sum((self.sum_person.values - forecast["yhat"].values) ** 2) / train_len)
        return rmse
    def predict(self, x):
        forecast = self.model.predict(pd.DataFrame({'ds':x}))
        self.model.plot(forecast)
        plt.title('全校保研人数')

class postgraduate_recommendation_B:
    def __init__(self, contest_path:str, data_path:str):
        self.data = self.get_data_B(contest_path, data_path)

    #获取去年B保数据
    def get_data_B(self, contest_path:str, data_path:str):
        #从竞赛表格pdf获取多少种竞赛种类
        pdf =  pdfplumber.open(contest_path) 
        table = []
        for pages in pdf.pages:
            for i in pages.extract_table()[1:]:
                table.append(i[1])
        #获取去年B保数据
        data_2023B = pd.read_excel(data_path)
        #计算每个竞赛有多少人参加
        ret = []
        for i in table:
            tmp = 0
            for j in data_2023B['附加分项目'].dropna().values:
                if j.find(i) != -1: tmp += 1
            ret.append(tmp)
        return pd.Series(ret, index=table)
    
    #生成词云
    def wordcloud_img(self):
        font = r'C:\Windows\Fonts\simfang.ttf' #显示中文字体
        wordcloud_ = wordcloud.WordCloud(font_path=font, height=600, width=1000, background_color='white')
        wordcloud_.fit_words(self.data.to_dict())
        return wordcloud_.to_image()
    
    #展示每个比赛参加人数
    def bar_img(self):
        plt.rc('font', family = 'SimHei', size = 9)
        plt.rc('axes', unicode_minus = False)
        plt.figure(figsize=(15, 10))
        self.data[self.data != 0].sort_values().plot.barh()