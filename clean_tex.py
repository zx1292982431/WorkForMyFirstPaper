import re
def calculate_chinese_ratio(sentence):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配中文字符的正则表达式
    total_characters = len(sentence)
    chinese_characters = re.findall(chinese_pattern, sentence)
    chinese_count = len(chinese_characters)
    chinese_ratio = chinese_count / total_characters
    return chinese_ratio

def replace_latex_syntax(text):
    # 定义要替换的正则表达式模式
    pattern = r'\$([^$]+)\$'

    # 使用re.sub()函数进行替换
    replaced_text = re.sub(pattern, 'TEX', text)

    return replaced_text


tex_files = [
    './tex/1/2023_06_01_36125522c510d15202afg.tex',
    './tex/2/2023_06_01_03cd5562f91812db75eag.tex',
    './tex/3/2023_06_01_f66e4a6c00a9938a85f7g.tex',
    './tex/4/2023_06_01_256e7660788135679d61g.tex',
    './tex/5/2023_06_01_5a926bb37eb5f84bb083g.tex',
    './tex/6/2023_06_01_db05bc9d26bdc21f6006g.tex',
    './tex/7/2023_06_01_848d4529194a1697eb96g.tex',
    './tex/8/2023_06_01_87dfe6bc0375db02bfb7g.tex',
    './tex/9/2023_06_01_98dd78cd49f29b7e3effg.tex',
    './tex/10/2023_06_01_98bd4484fcfac0cdda7ag.tex',
]

# tex_files = [
# "./tex_before/0--60/0--5/0--5.tex",
# "./tex_before/0--60/5--10/5--10.tex",
# "./tex_before/0--60/10--15/10--15.tex",
# "./tex_before/0--60/15--20/15--20.tex",
# "./tex_before/0--60/20--25/20--25.tex",
# "./tex_before/0--60/25--30/25--30.tex",
# "./tex_before/0--60/30--35/30--35.tex",
# "./tex_before/0--60/35--40/35--40.tex",
# "./tex_before/0--60/40--45/40--45.tex",
# "./tex_before/0--60/45--50/45--50.tex",
# "./tex_before/0--60/50--55/50--55.tex",
# "./tex_before/0--60/55--60/55--60.tex",
# "./tex_before/60--120/60--65/60--65.tex",
# "./tex_before/60--120/65--70/65--70.tex",
# "./tex_before/60--120/70--75/70--75.tex",
# "./tex_before/60--120/75--80/75--80.tex",
# "./tex_before/60--120/80--85/80--85.tex",
# "./tex_before/60--120/85--90/85--90.tex",
# "./tex_before/60--120/90--95/90--95.tex",
# "./tex_before/60--120/95--100/95--100.tex",
# "./tex_before/60--120/100--105/100--105.tex",
# "./tex_before/60--120/105--110/105--110.tex",
# "./tex_before/60--120/110--115/110--115.tex",
# "./tex_before/60--120/115--120/115--120.tex",
# "./tex_before/120--180/120--125/120--125.tex",
# "./tex_before/120--180/125--130/125--130.tex",
# "./tex_before/120--180/130--135/130--135.tex",
# "./tex_before/120--180/135--140/135--140.tex",
# "./tex_before/120--180/140--145/140--145.tex",
# "./tex_before/120--180/145--150/145--150.tex",
# "./tex_before/120--180/150--155/150--155.tex",
# "./tex_before/120--180/155--160/155--160.tex",
# "./tex_before/120--180/160--165/160--165.tex",
# "./tex_before/120--180/165--170/165--170.tex",
# "./tex_before/120--180/170--175/170--175.tex",
# "./tex_before/120--180/175--180/175--180.tex",
# "./tex_before/180--240/180--185/180--185.tex",
# "./tex_before/180--240/185--190/185--190.tex",
# "./tex_before/180--240/190--195/190--195.tex",
# "./tex_before/180--240/195--200/195--200.tex",
# "./tex_before/180--240/200--205/200--205.tex",
# "./tex_before/180--240/205--210/205--210.tex",
# "./tex_before/180--240/210--215/210--215.tex",
# "./tex_before/180--240/215--220/215--220.tex",
# "./tex_before/180--240/220--225/220--225.tex",
# "./tex_before/180--240/225--230/225--230.tex",
# "./tex_before/180--240/230--235/230--235.tex",
# "./tex_before/180--240/235--240/235--240.tex",
# "./tex_before/240--290/240--245/240--245.tex",
# "./tex_before/240--290/245--250/245--250.tex",
# "./tex_before/240--290/250--255/250--255.tex",
# "./tex_before/240--290/255--260/255--260.tex",
# "./tex_before/240--290/260--265/260--265.tex",
# "./tex_before/240--290/265--270/265--270.tex",
# "./tex_before/240--290/270--275/270--275.tex",
# "./tex_before/240--290/275--280/275--280.tex",
# "./tex_before/240--290/280--285/280--285.tex",
# "./tex_before/240--290/285--290/285--290.tex",
# ]

for tex_file in tex_files:
    with open(tex_file,encoding='utf-8') as file:
        for line in file:
            line = line.replace('\\begin{CJK}{UTF8}{mj}','')
            line = line.replace('\\end{CJK}','')
            line = line.replace('\\item','')
            line = line.replace('\\section','')
            line = line.replace('\\title','')
            line = line.replace(' ','')
            line = replace_latex_syntax(line)
            line.strip()
            if calculate_chinese_ratio(line)>=0.3:
                with open('data/cleaned_tex.tex', 'a',encoding='utf-8') as ans:
                    ans.write(line)