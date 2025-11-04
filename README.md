# BookRecommendation(CS5841 Group Project)
本项目使用了三个核心 CSV 文件：

- **Users.csv:** 包含用户信息 (User-ID, Location, Age)。
- **Books.csv:** 包含书籍信息 (ISBN, Book-Title, Book-Author, Year-Of-Publication 等)。
- **Ratings.csv:** 包含用户对书籍的评分 (User-ID, ISBN, Book-Rating)。



## 数据处理与预处理流程

### 1. `Ratings.csv` (评分数据) 处理



- **区分隐式与显式评分：** `Book-Rating` (书籍评分) 列包含 1-10 的“显式”评分和 `0` 的“隐式”评分（代表用户有过互动，但未打分）。
- **过滤数据：** 为了专注于有明确用户偏好（喜欢或不喜欢）的数据，我们**移除了所有 `Book-Rating` 为 0 的行**。模型将仅基于 1-10 的显式评分进行构建。



### 2. `Users.csv` (用户数据) 处理



- **分析 `Age` (年龄) 列：** 经分析，`Age` 列存在超过 39% 的大量缺失值 (NaN)。
- **决策：** 使用均值或中位数来填充如此大比例的缺失数据会引入严重的统计偏差，使该特征变得不可靠。
- **操作：** 我们选择**完全删除 (drop) `Age` 列**。



### 3. `Books.csv` (书籍数据) 处理



- **移除无关列：** `Image-URL-S`, `Image-URL-M`, 和 `Image-URL-L` 列对于协同过滤模型是无用的。这些列被**删除 (drop)** 以保持 DataFrame 的整洁。
- **清洗 `Year-Of-Publication` (出版年份)：**
  1. 首先，使用 `pd.to_numeric` 将该列强制转换为数字，无法转换的字符串（脏数据）会被设为 `NaN`。
  2. 然后，我们**不使用**填充（如中位数）来处理 `NaN` 或无效年份（如 `0` 或 `2050`）。
  3. 我们**只保留**了 `Year-Of-Publication` 在一个合理范围（例如 1900 年至 2025 年）内的**行**，所有包含无效或 `NaN` 年份的行都被**移除**，以确保数据的高质量。
- **填充文本空值：** 对于 `Book-Author` 和 `Publisher` 中剩余的少量 `NaN` 值，我们使用 `'Unknown'` 占位符进行了填充。



### 5. 模型准备：合并与稀疏性处理 (Merging & Sparsity Reduction)



- **合并数据：** 将上一步骤中清理干净的 `explicit_ratings` (显式评分表) 和 `books` (书籍表) 通过 `ISBN` 进行合并，生成主数据表 `final_df`。
- **解决数据稀疏性：**
  - **策略：** 我们通过设置阈值来**过滤数据**，只保留“活跃用户”和“热门书籍”。
  - **操作：**
    1. **书籍过滤：** 只保留那些评分总数**大于等于 100** 的书籍 (`MIN_BOOK_RATINGS = 100`)。
    2. **用户过滤：** 只保留那些评分总次数**大于等于 50** 的用户 (`MIN_USER_RATINGS = 50`)。
- **特征编码 (Feature Encoding)：**
  - 为了让模型能够处理分类数据，我们使用 `sklearn.preprocessing.LabelEncoder` 将 `User-ID` 和 `Book-Title` 转换为了连续的整数索引（`user_encoded` 和 `book_encoded`）。

