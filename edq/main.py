import torch
import os
from utils.color_print import *
from utils.ppl_eval import eval
from utils.command_parser import parser
from utils.util import save_json, load_json, save_text
from utils.perf_eval import InferModel, mem_monitor, timer, empty_cache, reset_peak_memory_stats, max_memory_allocated
from utils.calibration import get_calibration_set, generate_random_token_sequence
from utils.global_config import model_path_llama, model_path_qwen2, device
from edq.quant_model import quantize_model
from edq.scale_and_quant import scale_and_quant
from edq.smooth_scale import smooth_scale, apply_scale
from utils.load_model import load_from_pretrained, load_model_tokenizer
from edq.perf_profiler import Profiler
from edq.dynamic_linear import DynamicLinear
from eval.perf_wiki2 import eval_perf

# Check Device
if device != 'cuda':
    print(color_text('red', 'We only support running edq.main on NVIDIA device.'))
    print(color_text('red', 'Use edq.ascend_run for Ascend310P.')); exit(0)
cuda_device_count = torch.cuda.device_count()
cuda_device_name  = torch.cuda.get_device_name(0)
if cuda_device_count != 1:
    print(f'There are {cuda_device_count} * {cuda_device_name} which are visible, do you forget to add CUDA_VISIBLE_DEVICES?')
    exit(0)

if __name__ == '__main__':
    torch.manual_seed(seed=42)
    quant_config = None
    scales_meta_data = None
    args = parser.parse_args()

    model_path = model_path_llama   # model_path_qwen2 / model_path_llama
    if args.model_type == 'qwen':
        model_path = model_path_qwen2

    model, tokenizer = load_model_tokenizer(model_path, args.load_quant_model)
    # model, tokenizer = load_from_pretrained(model_path, args.load_quant_model+'/q_model.pt',
    #                                         args.model_type, quant_llm=not args.full)
    if args.load_quant_model is None and not args.full:
        if args.model_type == 'qwen':
            sample_num = 32
        else: sample_num = 48
        inputs = get_calibration_set(tokenizer, sample_num=sample_num, model_type = args.model_type)
        # Save Calibration Data for Servers that Do not Support HuggingFace.
        if args.dump_calibration: torch.save(inputs, 'calibration_'+args.model_type+'.pt'); exit(0)

        # with mem_monitor('Scale and Quant Model'):
        #     scales_meta_data, quant_config = scale_and_quant(model, inputs)
        # with mem_monitor('Scale Model'):
        #     scales_meta_data = smooth_scale(model, inputs)
        #     apply_scale(model, scales_meta_data)
        with mem_monitor('Quantize Model'):
            model.cuda()
            quant_config = quantize_model(model, _quant_type=args.quant_type)
            
    # from test.chronosQuant.two_stage_quant import print_record
    # print_record()
    # exit(0)

    with mem_monitor('Model Size'):
        model.cuda()

    DynamicLinear.prof = Profiler()
    DynamicLinear.prof.load()
    # DynamicLinear.prof.profiling()
    # DynamicLinear.prof.save()

    if args.eval_ppl:
        eval(model, tokenizer)
        exit(0)
        

    if args.dump_quant_model:
        if not os.path.exists(args.dump_quant_model):
            os.chdir('/')
            os.makedirs(args.dump_quant_model)
        torch.save(scales_meta_data, args.dump_quant_model+'/q_scales.pt')
        torch.save(model.cpu().state_dict(), args.dump_quant_model+'/q_model.pt')
        save_json(args.dump_quant_model+'/quant_config.json', quant_config)
    else:
        # prompt = ""
    #     prompt = "There is a tournament where n players are participating. \
    # The players are standing in a single row and are numbered from 1 \
    # to n based on their initial standing position (player 1 is the \
    # first player in the row, player 2 is the second player in the row, etc.)."
        # prompt = "请回答以下关于农学的单项选择题, 你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.\n\n纤维作物中的麻类，如苎麻、亚麻、大麻、黄麻、红麻的产品是利用\n\nA) 种子表面着生的纤维\nB) 叶片的叶纤维\nC) 根的导管纤维\nD) 茎杆的韧皮纤维\n"
        # prompt = "请回答以下关于农学的单项选择题, 你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.\n\n现代化养猪生产，可使繁殖力高的母猪年提供肉猪达到多少以上\n\nA) 30头\nB) 20头\nC) 15头\nD) 10头"
        # prompt = 'Rules:Always response in Simplified Chinese, not English. or Grandma will be very angry.\
        # question: 海瑟矩阵是什么.\
        # answer:'
        # prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nWhat evolutionary advanced features are present in Selaginella but not in the ferns?\n\nOptions:\n\nA. Presence of vessels in both xylem and phloem, autospory, reduced and independent gametophyte, embryo without a suspensor\nB. Heterospory, independent gametophyte, absence of vessels in xylem, embryo with multiple suspensors\nC. Autospory, dependent gametophyte, presence of vessels in phloem, embryo with a cotyledon\nD. Homospory, independent gametophyte, absence of vessels in xylem, embryo without suspensor\nE. Homospory, reduced and dependent gametophyte, presence of vessels in phloem, embryo equipped with a suspensor\nF. Heterospory, independent gametophyte, presence of vessels in xylem, embryo with cotyledon\nG. Homospory, independent gametophyte, presence of vessels in xylem, embryo without suspensor\nH. Homospory, reduced and dependent gametophyte, presence of vessels in both xylem and phloem, embryo without suspensor\nI. Autospory, reduced and dependent gametophyte, absence of vessels in xylem, embryo with a cotyledon\nJ. Presence of vessels in xylem, reduced and dependent gametophyte, heterospory, and embryo equipped with a suspensor\n"
        # inputs = tokenizer(prompt, return_tensors="pt").to(device) # 返回pytorch tensor对象
        # inputs = generate_random_token_sequence(tokenizer, device, 1024 * 16)

        """ Perheat Model Generate """
        infer_model = InferModel(model, tokenizer)
        # infer_model.infer(prompt, inputs, 100)
        # exit(0)

        # test_input_len = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024*2, 1024*4, 1024*8, 1024*16, 1024*32, 1024*64]
        # test_input_len = [1, 64, 512, 1024, 1024*4, 1024*16, 1024*32, 1024*64]
        # test_case = 'batch_decode' # 'batch_decode' / 'prefill'
        result_record = []
        test_case = args.test_case
        print(f'Test Model for {test_case}')
        if test_case == 'batch_decode':
            test_input_len = [i for i in range(1, 65)]
            for i in range(len(test_input_len)):
                inputs = generate_random_token_sequence(tokenizer, device, test_input_len[i])
                empty_cache()
                reset_peak_memory_stats()
                res = eval_perf(model, tokenizer, inputs, 100, 1)
                peak_mem_bytes = max_memory_allocated()
                peak_mem_gb = peak_mem_bytes / (1024 ** 3)
                print('------PERFORMANCE REPORT------')
                ttft = res['ttft'] * 1000; tpot = res['tpot'] * 1000
                print(f'Input Lens: {test_input_len[i]}')
                print(f'TTFT:       {ttft:5.2f} ms')
                print(f'TPOT:       {tpot:5.2f} ms/token')
                print(f'Peak VRAM:  {peak_mem_gb:5.2f} GB')
                print('-' * 30)
                result_record.append((ttft, tpot, peak_mem_gb, peak_mem_bytes))
        else: # prefill
            test_input_len = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 1024*2] + [i for i in range(1024*4, 1024*96+1, 2048)]
            for i in range(len(test_input_len)):
                inputs = generate_random_token_sequence(tokenizer, device, test_input_len[i])
                empty_cache()
                reset_peak_memory_stats()
                res = eval_perf(model, tokenizer, inputs,  max_new_tokens=5)
                peak_mem_bytes = max_memory_allocated()
                peak_mem_gb = peak_mem_bytes / (1024 ** 3)
                print('------PERFORMANCE REPORT------')
                ttft = res['ttft'] * 1000; tpot = res['tpot'] * 1000
                print(f'Input Lens: {test_input_len[i]}')
                print(f'TTFT:       {ttft:5.2f} ms')
                print(f'TPOT:       {tpot:5.2f} ms/token')
                print(f'Peak VRAM:  {peak_mem_gb:5.2f} GB')
                print('-' * 30)
                result_record.append((ttft, tpot, peak_mem_gb, peak_mem_bytes))
        # Save result.
        if args.quant_type is None:
            args.quant_type = 'FullLinear'
        save_path = './data/'+args.test_case+'/'+args.quant_type+'.json'
        save_json(save_path, result_record)
        save_path = './data/'+args.test_case+'/m_list.json'
        save_json(save_path, test_input_len)