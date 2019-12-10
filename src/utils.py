import os
import torch
import pickle

def save_model_by_name(model, optimizer, tokenizer, model_save_name, global_step):
    save_dir = os.path.join('..', 'models', 'checkpoints', model_save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'tokenizer': tokenizer}
    torch.save(state, file_path)

def save_joint_by_name(generator, predictor, optimizer, tokenizer, model_save_name, global_step):
    print("saving joint")
    save_dir = os.path.join('..', 'models', 'checkpoints', model_save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = {'generator_dict': generator.state_dict(),
             'predictor_dict': predictor.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'tokenizer': tokenizer}
    torch.save(state, file_path)

def save_losses_by_name(losses, model_save_name, global_step):
    save_dir = os.path.join('..', 'models', 'checkpoints', model_save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'losses-{:05d}.pkl'.format(global_step))
    with open(file_path, 'wb') as f:
        pickle.dump(losses, f)
    f.close()

def create_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    return mask
