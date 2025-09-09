
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Tuple, Dict, Set
import warnings
warnings.filterwarnings('ignore')

class SRComplexityCalculator:
    """计算符合PySR标准的复杂度"""
    
    def __init__(self):
        self.binary_ops = ['+', '-', '*', '/']
        self.unary_functions = ['exp', 'log', 'sqrt', 'inv', 'square']
        self.variables = ['t', 'C_eth', 'C_pg', 'C_pol']
    
    def find_constants(self, expr: str) -> Tuple[List[str], Set[int]]:
        """找到表达式中的所有常数并返回常数列表和它们占用的位置"""
        clean_expr = expr.replace(' ', '')
        constant_pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        
        constants = []
        positions = set()
        
        for match in re.finditer(constant_pattern, clean_expr):
            constant = match.group(0)
            start_idx = match.start()
            end_idx = match.end()
            
            # 检查是否不是变量名的一部分
            before = clean_expr[start_idx - 1] if start_idx > 0 else ''
            after = clean_expr[end_idx] if end_idx < len(clean_expr) else ''
            
            if not re.match(r'[a-zA-Z_]', before) and not re.match(r'[a-zA-Z_]', after):
                constants.append(constant)
                # 标记这个常数占用的所有位置
                for i in range(start_idx, end_idx):
                    positions.add(i)
        
        return constants, positions
    
    def count_operations(self, expr: str, constant_positions: Set[int]) -> Dict[str, int]:
        """计算运算符数量，排除常数中的负号"""
        clean_expr = expr.replace(' ', '')
        op_counts = {}
        
        for op in self.binary_ops:
            pattern = '\\' + op
            op_count = 0
            for match in re.finditer(pattern, clean_expr):
                if match.start() not in constant_positions:
                    op_count += 1
            if op_count > 0:
                op_counts[op] = op_count
                
        return op_counts
    
    def count_functions(self, expr: str) -> Dict[str, int]:
        """计算函数数量"""
        func_counts = {}
        for func in self.unary_functions:
            pattern = func + r'\('
            matches = re.findall(pattern, expr)
            if matches:
                func_counts[func] = len(matches)
        return func_counts
    
    def count_variables(self, expr: str) -> Dict[str, int]:
        """计算变量出现次数（包括重复）"""
        var_counts = {}
        for var in self.variables:
            pattern = r'\b' + re.escape(var) + r'\b'
            matches = re.findall(pattern, expr)
            if matches:
                var_counts[var] = len(matches)
        return var_counts
    
    def calculate_complexity(self, expr: str) -> Tuple[int, Dict]:
        """计算表达式复杂度并返回详细分解"""
        if pd.isna(expr) or expr == '':
            return 0, {}
        
        complexity = 0
        breakdown = {}
        
        # 1. 找到常数
        constants, constant_positions = self.find_constants(expr)
        if constants:
            breakdown['constants'] = constants
            complexity += len(constants)
        
        # 2. 计算运算符（排除常数中的负号）
        op_counts = self.count_operations(expr, constant_positions)
        if op_counts:
            breakdown['operations'] = op_counts
            complexity += sum(op_counts.values())
        
        # 3. 计算函数
        func_counts = self.count_functions(expr)
        if func_counts:
            breakdown['functions'] = func_counts
            complexity += sum(func_counts.values())
        
        # 4. 计算变量
        var_counts = self.count_variables(expr)
        if var_counts:
            breakdown['variables'] = var_counts
            complexity += sum(var_counts.values())
        
        return complexity, breakdown

class ParetoFrontAnalyzer:
    """Pareto前沿分析器"""
    
    @staticmethod
    def is_dominated(point1: Tuple[float, float], point2: Tuple[float, float]) -> bool:
        """检查point1是否被point2支配（假设两个目标都是最小化）"""
        complexity1, loss1 = point1
        complexity2, loss2 = point2
        
        # point2支配point1当且仅当：
        # point2在两个目标上都不差于point1，且至少在一个目标上严格更好
        return (complexity2 <= complexity1 and loss2 <= loss1 and 
                (complexity2 < complexity1 or loss2 < loss1))
    
    @staticmethod
    def find_pareto_front(data: pd.DataFrame, complexity_col: str = 'restored_complexity', 
                         loss_col: str = 'loss') -> pd.DataFrame:
        """找到Pareto前沿"""
        pareto_indices = []
        
        for i in range(len(data)):
            is_pareto = True
            point_i = (data.iloc[i][complexity_col], data.iloc[i][loss_col])
            
            for j in range(len(data)):
                if i != j:
                    point_j = (data.iloc[j][complexity_col], data.iloc[j][loss_col])
                    if ParetoFrontAnalyzer.is_dominated(point_i, point_j):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return data.iloc[pareto_indices].sort_values(complexity_col)

class SRDataProcessor:
    """主要的数据处理类"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.calculator = SRComplexityCalculator()
        self.analyzer = ParetoFrontAnalyzer()
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"成功加载数据，共{len(self.data)}行")
            print(f"列名: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"文件未找到: {self.file_path}")
            return None
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def calculate_complexities(self) -> None:
        """计算restored_equation的复杂度"""
        if self.data is None:
            print("请先加载数据")
            return
        
        if 'restored_equation' not in self.data.columns:
            print("数据中没有'restored_equation'列")
            return
        
        complexities = []
        breakdowns = []
        
        print("正在计算复杂度...")
        for idx, row in self.data.iterrows():
            equation = row['restored_equation']
            complexity, breakdown = self.calculator.calculate_complexity(equation)
            complexities.append(complexity)
            breakdowns.append(breakdown)
            
            if idx % 10 == 0:
                print(f"已处理 {idx + 1}/{len(self.data)} 个方程")
        
        self.data['restored_complexity'] = complexities
        self.data['complexity_breakdown'] = breakdowns
        print("复杂度计算完成")
    
    def analyze_pareto_front(self) -> pd.DataFrame:
        """分析Pareto前沿"""
        if 'restored_complexity' not in self.data.columns:
            print("请先计算复杂度")
            return None
        
        pareto_data = self.analyzer.find_pareto_front(self.data)
        print(f"找到 {len(pareto_data)} 个Pareto前沿解")
        return pareto_data
    
    def print_complexity_details(self, num_examples: int = 5) -> None:
        """打印复杂度计算详情"""
        if self.data is None or 'complexity_breakdown' not in self.data.columns:
            print("请先计算复杂度")
            return
        
        print("\n=== PySR复杂度计算规则 ===")
        print("• 每个运算操作 (+, -, *, /) = 1分")
        print("• 每个函数 (exp, log, sqrt, inv, square) = 1分")
        print("• 每个常数 = 1分 (如 -63.809753 = 1分)")
        print("• 每个变量出现 = 1分 (重复计算)")
        print("• 常数中的负号不算作运算符")
        
        print(f"\n=== 前{num_examples}个模型的复杂度详情 ===")
        
        for i in range(min(num_examples, len(self.data))):
            row = self.data.iloc[i]
            print(f"\nModel {i+1}: {row['restored_equation']}")
            
            breakdown_parts = []
            total_calc = 0
            
            breakdown = row['complexity_breakdown']
            
            if 'constants' in breakdown:
                constants = breakdown['constants']
                breakdown_parts.append(f"constants{constants}({len(constants)})")
                total_calc += len(constants)
            
            if 'operations' in breakdown:
                ops = breakdown['operations']
                for op, count in ops.items():
                    breakdown_parts.append(f"{op}({count})")
                    total_calc += count
            
            if 'functions' in breakdown:
                funcs = breakdown['functions']
                for func, count in funcs.items():
                    breakdown_parts.append(f"{func}({count})")
                    total_calc += count
            
            if 'variables' in breakdown:
                vars_dict = breakdown['variables']
                var_parts = [f"{var}({count})" for var, count in vars_dict.items()]
                breakdown_parts.append(f"vars[{','.join(var_parts)}]({sum(vars_dict.values())})")
                total_calc += sum(vars_dict.values())
            
            breakdown_str = " + ".join(breakdown_parts) + f" = {total_calc}"
            print(f"Breakdown: {breakdown_str}")
            print(f"Calculated: {row['restored_complexity']} | Original PySR: {int(row['complexity'])}")
    
    def create_visualization(self, save_path: str = None) -> None:
        """创建可视化图表"""
        if 'restored_complexity' not in self.data.columns:
            print("请先计算复杂度")
            return
        
        # 找到Pareto前沿
        pareto_data = self.analyze_pareto_front()
        
        # 创建图表
        # plt.figure(figsize=(12, 8))
        
        # 绘制所有点
        plt.scatter(self.data['restored_complexity'], self.data['loss'], 
                   alpha=1, s=50, c='lightblue', label='All Models')
        
        # 绘制Pareto前沿
        plt.scatter(pareto_data['restored_complexity'], pareto_data['loss'], 
                   alpha=0.8, s=100, c='gold', marker='*', 
                   edgecolors='red', linewidth=1, label='Pareto Front')
        
        # 连接Pareto前沿点
        pareto_sorted = pareto_data.sort_values('restored_complexity')
        plt.plot(pareto_sorted['restored_complexity'], pareto_sorted['loss'], 
                'r--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Restored Complexity')
        plt.ylabel('Loss')
        # plt.title('Complexity vs Loss (Pareto Front Analysis)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def export_results(self, output_path: str) -> None:
        """导出结果"""
        if self.data is None:
            print("没有数据可导出")
            return
        
        # 准备导出数据
        export_data = self.data.copy()
        
        # 将breakdown转换为字符串以便导出
        if 'complexity_breakdown' in export_data.columns:
            export_data['complexity_breakdown_str'] = export_data['complexity_breakdown'].astype(str)
            export_data = export_data.drop('complexity_breakdown', axis=1)
        
        # 添加是否为Pareto前沿的标记
        pareto_data = self.analyze_pareto_front()
        export_data['is_pareto_front'] = export_data.index.isin(pareto_data.index)
        
        export_data.to_csv(output_path, index=False)
        print(f"结果已导出至: {output_path}")
    
    def print_summary(self) -> None:
        """打印分析摘要"""
        if self.data is None:
            return
        
        pareto_data = self.analyze_pareto_front()
        
        print("\n=== 分析摘要 ===")
        print(f"总模型数: {len(self.data)}")
        print(f"Pareto前沿模型数: {len(pareto_data)}")
        print(f"平均复杂度 (restored): {self.data['restored_complexity'].mean():.1f}")
        print(f"最佳损失: {self.data['loss'].min():.6f}")
        print(f"复杂度范围: {self.data['restored_complexity'].min()} - {self.data['restored_complexity'].max()}")
        
        print("\n=== Pareto前沿模型 ===")
        for idx, (_, row) in enumerate(pareto_data.iterrows()):
            print(f"{idx+1}. 复杂度: {row['restored_complexity']}, 损失: {row['loss']:.6f}")
            print(f"   方程: {row['restored_equation']}")

def main():
    """主函数"""
    # 文件路径
    for i in range(1, 9):  # 处理run-1,- , run-8
        file_path = fr"Symbolic Regression/srloop/data/hall_of_fame_run-{i}_restored.csv"

        # 创建处理器
        processor = SRDataProcessor(file_path)
        
        # 加载数据
        data = processor.load_data()
        if data is None:
            return
        
        # 计算复杂度
        processor.calculate_complexities()
        
        # 打印复杂度详情
        processor.print_complexity_details(num_examples=10)
        
        # 分析并打印摘要
        processor.print_summary()
        
        # 创建可视化
        processor.create_visualization(save_path=f"Symbolic Regression/srloop/visualisation/pareto_front_analysis_run-{i}.png")

        # 导出结果
        output_path = file_path.replace('.csv', '_analyzed.csv')
        processor.export_results(output_path)

if __name__ == "__main__":
    main()