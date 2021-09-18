"""
公式生成器
"""

# 开始行、结束行
(START_ROW, END_ROW) = (4, 52)
# 开始列、结束列
(START_COL, END_COL) = ('C', 'N')
# 下周计划列
NEXT_PLAN_COL = 'O'


def gen_this_week_data():
    """
    生成本周数据，元组（日期, 分类, 明细)
    """
    head_row = f"""data=[];"""
    # 获取
    def day_group(row_num):
        """
        按照日期列分组（2个一组）
        :param row_num 行号
        :return: 从开始列，结束列的所有分组（日期单元格,分类列,明细列)
        """
        start_col_ord = ord(START_COL)
        end_col_ord = ord(END_COL)
        col_ord = start_col_ord

        data = []
        while col_ord < end_col_ord:
            # 一次取三个列
            data.append((f'{chr(col_ord)}2',  f'{chr(col_ord)}{row_num}', f'{chr(col_ord + 1)}{row_num}'))
            col_ord += 2

        return data

    def concat_groups(groups):
        """
        拼接所有列到列表
        :param group: 三元组
        """
        return '"data.append([' \
        + ','.join([ f"""('" & TEXT({group[0]},"mm-dd") & "','" & {group[1]} & "','" & {group[2]} & "')""" for group in groups ]) \
        + '])"'

    all_groups = list([ concat_groups(day_group(row_num)) for row_num in range(START_ROW, END_ROW)])
    all_groups[0] = f'="data=[];" & {all_groups[0]}'

    return "\n=".join(all_groups)


def gen_next_week_data():
    """
    生成下周数据
    """
    next_data = list([ f""" "next_data.append('" & {NEXT_PLAN_COL}{row_num} & "')" """ for row_num in range(START_ROW, END_ROW)])
    next_data[0] = f'="next_data = [];" & {next_data[0]}'

    return "\n=".join(next_data)


if __name__ == '__main__':
    print("-" * 100)
    print("- 日报数据")
    print(gen_this_week_data())
    print("-" * 100)
    print("- 下周数据")
    print(gen_next_week_data())