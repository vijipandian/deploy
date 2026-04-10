import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from google.colab import files
uploaded=files.upload()

users=pd.read_csv("Users.csv")
st.write(users.head(2))

teacher=pd.read_csv("Teachers.csv")
st.write(teacher.head(2))

course=pd.read_csv("Courses.csv")
st.write(course.head(2))

transcation=pd.read_csv("Transactions.csv")
st.write(transcation.head(2))

data=(transcation.merge(teacher,on="TeacherID",how="left")\
                .merge(course,on="CourseID",how="left")\
                .merge(users,on="UserID",how="left")
)
data.head(2)

data.info()
data.drop_duplicates(inplace=True)
data.describe()
data.isna().sum()

st.title("Problem Statement")
st.title("1.Which instructors consistently deliver high-quality courses?")

quality=data.groupby(
    ["TeacherID","TeacherName"]
).agg(
    Avg_Teacher_rating=("TeacherRating","mean"),
    Avg_Course_rating=("CourseRating","mean"),
    Total_Enrollments=("TransactionID","count"),
    Experience=("YearsOfExperience","mean")
).reset_index()
quality.head()


top=quality.sort_values(
    by="Avg_Teacher_rating",
    ascending=False
)
top.head(5)


fig,ax=plt.subplots()
sns.barplot(
    data=top.head(10),
    x="Avg_Teacher_rating",
    y="TeacherName",
    palette='Set2',
    hue="Experience",
    ax=ax
)
st.title(":red[Top 10 High Quality Instructors]")
ax.set_xlabel("Average Teacher Rating")
ax.set_ylabel("TeacherName")
st.pyplot(fig)

st.title("2. Does teaching experience translate into better-rated courses?")
fig,ax=plt.subplots()
sns.scatterplot(data=data,x="YearsOfExperience",y="CourseRating",color="purple",ax=ax)
st.title(":red[Teaching Experience vs Course Rating]")
ax.set_xlabel("Years of Teaching Experience")
ax.set_ylabel("Course Rating")
st.pyplot(fig)

st.title("3.Are some course categories more dependent on instructor quality?")

cate=data.groupby("CourseCategory")[["TeacherRating","CourseRating"]].mean()

fig,ax=plt.subplots()
sns.heatmap(cate,annot=True,cmap="YlGnBu",ax=ax)
st.title(":red[Course Category vs Quality]")
st.pyplot(fig)

fig,ax=plt.subplots()
sns.boxplot(data=data,x="CourseCategory",y="CourseRating",hue="CourseCategory",palette="Set1",ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title(":red[CourseRating vs Category]")
st.pyplot(fig)

st.title("4.How evenly is teaching performance distributed across the platform?")

fig,ax=plt.subplots()
sns.histplot(data=data,x="TeacherRating",bins=10,kde=True,ax=ax)
st.title("Distribution Of Teaching Performance")
ax.set_xlabel("Teacher Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

performance=data.groupby("TeacherName")[["YearsOfExperience","TeacherRating"]].mean().reset_index()
st.write(performance)

st.title(":red[Experience vs Rating Scatter Plot]")

fig,ax=plt.subplots()
sns.scatterplot(data=performance,x="YearsOfExperience",y="TeacherRating",color="purple",ax=ax)
st.title("Experience Vs Performance")
ax.set_xlabel("Years Of Experience")
ax.set_ylabel("Teacher Rating")
st.pyplot(fig)

st.title(":red[Top Performance]")
top=performance.sort_values(
    by="TeacherRating",
    ascending=False
)
st.write(top.head(5))

st.title("B.Key Analytical Question")
st.title("1. What is the overall distribution of instructor ratings?")
st.title(":green[HISTOGRAM(MAIN DISTRIBUTION PLOT)]")
fig,ax=plt.subplots()
sns.histplot(data["TeacherRating"],bins=10,kde=True,color="green",ax=ax)
st.title(":red[Distribution of Ratings]")
ax.set_xlabel("Teacher Rating")
ax.set_ylabel("Number of Instructors")
st.pyplot(fig)

st.title(":green[Boxplot(spread+outliers)]")
fig,ax=plt.subplots()
sns.boxplot(y=data["TeacherRating"],color="purple")
st.title(":red[Spread of Instructor Rating]")
st.pyplot(fig)


st.title("2.Do instructors with more experience receive higher ratings?")
data["RatingCategory"]=pd.cut(data["TeacherRating"],bins=[1.05,2.67,3.56,4.97],labels=["low","medium","high"])
data["RatingCategory"].value_counts()

res=data.groupby("YearsOfExperience")["TeacherRating"].mean().reset_index()
res

fig,ax=plt.subplots()
sns.lineplot(data=res,x="YearsOfExperience",y="TeacherRating",marker="o",color="orange")
st.title(":red[Experience vs Teacher Rating]")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("TeacherRating")
st.pyplot(fig)


st.title("3.Is there a relationship between TeacherRating and CourseRating?")

relation=data.groupby("TeacherRating")['CourseRating'].mean().reset_index()
relation

fig,ax=plt.subplots()
sns.lineplot(data=relation,x="TeacherRating",y="CourseRating",marker="o",color="orange")
st.title(":red[TeacherRating Vs CourseRating]")
ax.set_xlabel("TeacherRating")
ax.set_ylabel("CourseRating")
st.pyplot(fig)

cor=data[['CourseRating','TeacherRating']].corr()
st.write(cor)

st.title("4.Which expertise areas consistently deliver high-quality courses?")

expertise=data.groupby("Expertise")["CourseRating"].mean().reset_index()
expertise

top_expertise=expertise.sort_values(by="CourseRating",ascending=False)
st.write(top_expertise)

fig,ax=plt.subplots()
sns.barplot(data=expertise,x="Expertise",y="CourseRating",hue="Expertise",palette="Set1")
ax.set_title(":red[Expertise Vs CourseRating]")
ax.set_xlabel("Expertise")
ax.set_ylabel("CourseRating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title("5.Are highly rated instructors associated with higher enrollments?")

enroll=data.groupby("TeacherRating")["TransactionID"].count().reset_index()
enroll.head(5)

fig,ax=plt.subplots()
sns.scatterplot(data=enroll,x="TeacherRating",y="TransactionID",color="blue",ax=ax)
st.title(":red[TeacherRating Vs Enrollments]")
ax.set_xlabel("TeacherRating")
ax.set_ylabel("Enrollments")
st.pyplot(fig)


cor=data.groupby("TeacherID").agg({
    "TeacherRating":"mean",
    "TransactionID":"count"
    })
st.write(cor)  

st.title("C.Analytical Methodology (Step-by-Step)")
st.title("1.Data Integration")

st.title(":green[a.Join Teachers ↔ Courses ↔ Transactions using TeacherID and CourseID]\n [b. Validate mapping between instructors and their courses]")
data1=transcation.merge(teacher,on="TeacherID",how="left")\
                 .merge(course,on="CourseID",how="left")
st.write(data1.head(3))

st.title("View Teacher - Course Combinations.")
da=data[["TeacherName","CourseName","Expertise"]]
st.write(da.head(4))

data1.isna().sum()
data1.nunique()

df=data.groupby(["TeacherName","CourseName"]).size().reset_index(name="Enrollments")
st.write(df.head(3))

st.title("2.Instructor Profile Analysis")
st.title(" a.Distribution of instructor age, experience, and expertise")
st.title(":green[AGE Distribution]")

fig,ax=plt.subplots()
sns.histplot(data['Age_x'],bins=15,kde=True,color="orange",ax=ax)
st.title("Distribution of Age")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

st.title(":green[Experience Distibution]")
fig,ax=plt.subplots()
sns.histplot(data["YearsOfExperience"],bins=15,kde=True,color="purple",ax=ax)
st.title("Distribution of Experience")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Count")
st.pyplot(fig)

st.title(":green[Expertise]")
fig,ax=plt.subplots()
sns.countplot(data=data,x="Expertise",hue="Expertise",palette="Set1",ax=ax)
st.title("Distribution of Expertise")
ax.set_xlabel("Expertise")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)


st.title("b.Rating spread across instructors")
fig,ax=plt.subplots()
sns.histplot(data["TeacherRating"],bins=15,kde=True,color="green",ax=ax)
st.title("Distribution of TeahcerRating")
ax.set_xlabel("TeacherRating")
ax.set_ylabel("Count")
st.pyplot(fig)

st.title(":green[Rating Spred by Instructor and Visualization]")
spread=data.groupby("TeacherName")["TeacherRating"].mean().reset_index()
st.write(spread.head(2))

fig,ax=plt.subplots()
sns.boxplot(x=data["TeacherRating"],color="blue",ax=ax)
st.title("Spread of Rating")
st.pyplot(fig)

st.title(":green[Top Rating]")

st.write(spread.sort_values(by="TeacherRating",ascending=False).head(3))

st.title("Identification of top-performing and low-performing instructors")
per=data.groupby(["TeacherName","YearsOfExperience"])["TeacherRating"].mean().reset_index()
st.write(per.head(2))

st.title(":green[Top Performing Instructors VS Visualizations]")

top=per.sort_values(by="TeacherRating",ascending=False).head(10)
st.write(top.head(2))

fig,ax=plt.subplots()
sns.barplot(data=top,x="TeacherRating",y="TeacherName",hue="TeacherName",palette="Set1",ax=ax)
st.title("Top Performing Instructors")
ax.set_xlabel("Teacher Rating")
ax.set_ylabel("Instructor")
st.pyplot(fig)

st.title(":green[Low Performance vs Visualiation]")
low=per.sort_values(by="TeacherRating").head(10)
st.write(low.head(2))

fig,ax=plt.subplots()
sns.barplot(data=low,x="TeacherRating",y="TeacherName",hue="TeacherName",palette="Set1",ax=ax)
st.title("Low Performing Instructors")
ax.set_xlabel("Teacher Rating")
ax.set_ylabel("Instructor")
st.pyplot(fig)

st.title("3.Experience vs Performance Analysis")
st.title(":green[Correlation between:\n a.YearsOfExperience and TeacherRating]")
cor=data[["YearsOfExperience","TeacherRating"]].corr()
st.write(cor)

sns.heatmap(cor,annot=True,cmap="coolwarm",ax=ax)
st.title("Correlation Between Experience and Rating")
st.pyplot(fig)

st.title(":green[b.YearsOfExperience and CourseRating]")
co=data[["YearsOfExperience","CourseRating"]].corr()
st.write(co)

sns.heatmap(co,annot=True,cmap="plasma",ax=ax)
st.title("Correlation Between Experience and CourseRating")
st.pyplot(fig)

st.title("c.Identify diminishing returns or experience thresholds")
ex=data.groupby("YearsOfExperience")["TeacherRating"].mean().reset_index()
st.write(ex.head(2))

fig,ax=plt.subplots()
sns.lineplot(data=ex,x="YearsOfExperience",y="TeacherRating",marker="o",color="green",ax=ax)
st.title("Experience vs TeacherRating")
ax.set_xlabel("YearsOf Experience")
ax.set_ylabel("TeacherRating")
st.pyplot(fig)

fig,ax=plt.subplots()
sns.scatterplot(data=data,x="YearsOfExperience",y="TeacherRating",color="green",ax=ax)
st.title("Experience vs Rating Distribution")
st.pyplot(fig)

st.title("4.Course Quality Evaluation")
st.title(":green[A.Course Rating analysis by:\n a.Course Category]")

cat=data.groupby("CourseCategory")["CourseRating"].mean().reset_index()

fig,ax=plt.subplots()
sns.barplot(data=cat,x="CourseCategory",y="CourseRating",hue="CourseRating",palette="Set1",ax=ax)
st.title("CourseRating vs CourseCategory Analysis")
ax.set_xlabel("CouseCategory")
ax.set_ylabel("CourseRating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
st.pyplot(fig)

st.title(":green[b.CourseRating Analysis by Course Level]")
level=data.groupby("CourseLevel")["CourseRating"].mean().reset_index()

fig,ax=plt.subplots()
sns.barplot(data=level,x="CourseLevel",y="CourseRating",hue="CourseLevel",palette="Set1",ax=ax)
st.title("CourseLevel vs CourseRating Analysis")
ax.set_xlabel("CourseLevel")
ax.set_ylabel("CourseRating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
st.pyplot(fig)

st.title("B.Gender vs course level comparisons")
st.write(data[["Gender_x","CourseLevel"]].head(3))

fig,ax=plt.subplots()
sns.countplot(data=data,x="CourseLevel",hue="Gender_x",color="salmon",ax=ax)
st.title("Gender vs CourseLevel comparisons")
ax.set_xlabel("CourseLevel")
ax.set_ylabel("Number of Course")
st.pyplot(fig)

st.title("C.Identify categories with consistently high or low ratings")
st.title(":green[High Category And Lowest Category]")
high=data.groupby("CourseCategory")["CourseRating"].mean().reset_index()
st.write(high.sort_values(by="CourseRating",ascending=False).head(3))
st.write(high.sort_values(by="CourseRating").head(3))

fig,ax=plt.subplots()
sns.barplot(data=high,x="CourseCategory",y="CourseRating",hue="CourseRating",palette="Set1",ax=ax)
st.title("High Rating Category")
ax.set_xlabel("CourseCategory")
ax.set_ylabel("CourseRating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title("5.Instructor Impact on Course Success")
st.title("A.Compare course ratings for:\n a.High-rated instructors \n b.Low-rated\n c.mid-rated")
data["Instructor"]=pd.cut(
    data["TeacherRating"],
    bins=[0,3,4,5],
    labels=["Low Rated","Mid Rated","High Rated"]
)
st.write(data[["TeacherName","TeacherRating","Instructor"]].head())
rating=data.groupby("Instructor")["CourseRating"].mean().reset_index()
st.write(rating)

fig,ax=plt.subplots()
sns.barplot(data=rating,x="Instructor",y="CourseRating",hue="CourseRating",palette="Set1",ax=ax)
st.title("Instructor Impact on Course Success")
ax.set_xlabel("Instructor")
ax.set_ylabel("CourseRating")
st.pyplot(fig)

st.title("B.Enrollment volume comparison by instructor rating tier")

data["InstructorTier"]=pd.cut(
    data["TeacherRating"],
    bins=[0,3,4,5],
    labels=["Low Rated","Mid Rated","High Rated"]
)
enroll=data.groupby("InstructorTier")["TransactionID"].count().reset_index()
st.write(enroll)

fig,ax=plt.subplots()
sns.barplot(data=enroll,
            x="InstructorTier",
            y="TransactionID",
            hue="TransactionID",palette="Set1",ax=ax)
st.title("Enrollment volume by Instructor Rating Tier")
ax.set_xlabel("Instructior Rating Tier")
ax.set_ylabel("Number of Enrollments")
st.pyplot(fig)

fig,ax=plt.subplots()
sns.boxplot(data=data,
            x="InstructorTier",
            y="Amount",hue="InstructorTier",
            palette="Set1",ax=ax)
st.title("Enrollment Spending by Instructor Tier")
st.pyplot(fig)

st.title("6.Expertise-Based Performance Insights")
st.title("A. Instructor expertise vs course quality")
ex_qa=data.groupby("Expertise")["CourseRating"].mean().reset_index()
ex_qa

fig,ax=plt.subplots()
sns.barplot(x="Expertise",y="CourseRating",data=ex_qa,palette="Set1",hue="CourseRating",ax=ax)
st.title("Instructor expertise vs course quality")
ax.set_xlabel("Instructor Expertise")
ax.set_ylabel("Course Quality")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title("B.Identify domains where teaching quality is most critical")
domain=data.groupby("Expertise")[["TeacherRating","CourseRating"]].mean().reset_index()
domain
fig,ax=plt.subplots()
sns.scatterplot(data=domain,x="TeacherRating",y="CourseRating",hue="Expertise",s=100,ax=ax)
st.title("Teaching Quality Importance by Domain")
ax.set_xlabel("TeacherRating")
ax.set_ylabel("CourseRating")
st.pyplot(fig)

st.title(":green[HeadMap]")
pivot=data.pivot_table(values="CourseRating",index="Expertise",columns="CourseLevel",aggfunc="mean")
fig,ax=plt.subplots()
sns.heatmap(pivot,annot=True,cmap="YlGnBu",ax=ax)
st.title("Course Quality by Expertise and CourseLevel")
st.pyplot(fig)

st.title(":green[Top Domain and Low Domain]")
domain.sort_values(by="CourseRating",ascending=False)
domain.sort_values(by="CourseRating")
fig,ax=plt.subplots()
sns.scatterplot(data=data,x="TeacherRating",y="CourseRating",hue="Expertise",ax=ax)
st.title("CourseRating vs TeacherRating Vs Instructor Expertise")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title("C. Highlight expertise gaps or training needs")
exp_level=data.groupby(["Expertise","CourseLevel"])["CourseRating"].mean().reset_index()
exp_level

fig,ax=plt.subplots()
sns.barplot(data=exp_level,x="Expertise",y="CourseRating",hue="CourseLevel",ax=ax)
st.title("Expetise vs Course Rating Level")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title(":green[Expetise vs Course Duration]")
dur=data.groupby(["Expertise","CourseDuration"])["CourseRating"].mean().reset_index()
dur
fig,ax=plt.subplots()
sns.barplot(data=dur,x="Expertise",y="CourseRating",hue="CourseDuration",ax=ax)
st.title("Expertise vs Course Duration Rating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title(":purple[Expertise Gaps]")
gap=data.groupby(["Expertise","CourseLevel","CourseDuration"])["CourseRating"].mean().reset_index()
st.write(gap.sort_values("CourseRating").head(5))

st.title("Key Performance Indicators (KPIs)")
st.title("Average Teacher Rating")
avg_tec=data["TeacherRating"].mean()
st.title("Average Course Rating")
avg_cou=data["CourseRating"].mean()
st.title("Rating Consistency Index")
Cons=1-data["TeacherRating"].std()
st.title("Experience Imapct Score")
exp=data["YearsOfExperience"].corr(data["TeacherRating"])
st.title("Enrollment Influence Ratio")
en=data.groupby("TeacherID").agg({
    "TeacherRating":"mean",
    "TransactionID":"count"
}).reset_index()
en.columns=["TeacherID","avg_tec","EnrollmentCount"]

infu=en["avg_tec"].corr(en["EnrollmentCount"])
print(infu)


st.metric(":blue[Average Teacher Rating]",round(avg_tec,2))
st.metric(":violet[Avg Course Rating]",round(avg_cou,2))
st.metric(":orange[Consistency Index]",round(Cons,2))
st.metric(":red[Experience Imapct]",round(exp,2))
st.metric(":green[Enrollment Influence]",round(infu,2))

st.title("🚀 .Streamlit Web Application Requirements")
st.title("1.Instructor performance leaderboard")
leader=data.groupby("TeacherID").agg({
    "TeacherRating":"mean",
    "CourseRating":"mean",
    "TransactionID":"count"
}).reset_index()

leader.columns=["TeacherID","avg_tec","avg_cou","EnrollmentCount"]
leader["Rank"]=(leader["avg_tec"]*0.4+
                leader["avg_cou"]*0.4+
                leader["EnrollmentCount"]*0.2)
leader=leader.sort_values(by="Rank",ascending=False)                
leader.head()
st.title("🏆Instructor LeaderBorad")
st.dataframe(leader)


st.title("2.Experience vs rating scatter plots")
fig,ax=plt.subplots()
sns.scatterplot(data=data,x="YearsOfExperience",y="TeacherRating",ax=ax,color="green")
ax.set_xlabel("YearsOfExperience")
ax.set_ylabel("TeacherRating")
st.title("Experience Vs TeacherRating")
st.pyplot(fig)


st.title("3.Course Quality Heatmap")
pivot=data.pivot_table(values="CourseRating",index="CourseCategory",columns="CourseLevel",aggfunc="mean")
fix,ax=plt.subplots()
sns.heatmap(pivot,annot=True,cmap="YlGnBu",ax=ax)
st.title("CourseQuality Heatmap")
st.pyplot(fig)

st.title("4.Expertise Vs TeacherRating")
exp=data.groupby("Expertise").agg({"TeacherRating":"mean","CourseRating":"mean"}).reset_index()
exp

fig,ax=plt.subplots()
sns.barplot(data=exp,x="Expertise",y="TeacherRating",ax=ax)
ax.set_title("Expertise Vs TeacherRating")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
st.pyplot(fig)

st.title("B.User Capabilities")
st.title(":green[1.Instructor Expertise filter]")
#B.use vapabilities
#1.instructor expertise filtes
exper=data["Expertise"].unique()
expe=st.sidebar.multiselect("SelectnInstructor Expertise",options=exper,default=exper)
st.write(exp)
filte=data[data["Expertise"].isin(expe)]
st.write(filte)


st.title(":red[2.Course Category and Level selectors]")
cate=st.sidebar.multiselect("Select Course Category",
      options=data["CourseCategory"].unique(),
      default=data["CourseCategory"].unique()
      )
level=st.sidebar.multiselect("Select CourseLevel",
      options=data["CourseLevel"].unique(),
      default=data["CourseLevel"].unique()
      )

#fillter data
filt=data[(data["CourseCategory"].isin(cate))&
          (data["CourseLevel"].isin(level))
        ]

#Heatmap
pivot=filt.pivot_table(
    values="CourseRating",
    index="CourseCategory",
    columns="CourseLevel",
    aggfunc="mean"
)
sns.heatmap(pivot,annot=True,cmap="YlGnBu")
st.pyplot(plt)

st.title(":blue[3.Rating range sliders]")
tea=st.sidebar.slider("Select TeacherRating",
      float(data["TeacherRating"].min()),
      float(data["TeacherRating"].max()),
      (float(data["TeacherRating"].min()),float(data["TeacherRating"].max()))
      )
cour=st.sidebar.slider("Select CourseRating",
      float(data["CourseRating"].min()),
      float(data["CourseRating"].max()),
      (float(data["CourseRating"].min()),float(data["CourseRating"].max()))
      )
fil=data[data["TeacherRating"].between(tea[0],tea[1])&
         data["CourseRating"].between(cour[0],cour[1])
         ]
st.write(fil.head())        

