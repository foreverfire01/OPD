
def init():
    global _global_dict
    _global_dict = {}

    _global_dict['numshiptrue1'] = 0  
    _global_dict['numship1'] = 0  
    _global_dict['numtrue1'] = 0 

    _global_dict['numshiptrue2'] = 0  
    _global_dict['numship2'] = 0
    _global_dict['numtrue2'] = 0

    _global_dict['numshiptrue3'] = 0  
    _global_dict['numship3'] = 0 
    _global_dict['numtrue3'] = 0
     
    _global_dict['numshiptrue4'] = 0  
    _global_dict['numship4'] = 0
    _global_dict['numtrue4'] = 0

    _global_dict['picsum'] = 0         
    _global_dict['picship'] = 0       

    _global_dict['picshiptrue1'] = 0    
    _global_dict['picemptytrue1'] = 0  
    _global_dict['picselectasship1'] = 0     

    _global_dict['picshiptrue2'] = 0    
    _global_dict['picemptytrue2'] = 0 
    _global_dict['picselectasship2'] = 0         

    _global_dict['picshiptrue3'] = 0   
    _global_dict['picemptytrue3'] = 0 
    _global_dict['picselectasship3'] = 0       

    _global_dict['picshiptrue4'] = 0 
    _global_dict['picemptytrue4'] = 0 
    _global_dict['picselectasship4'] = 0  

    
def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    return _global_dict[key]


def add(key, num):
    _global_dict[key] += num

def reset():
    _global_dict['numshiptrue1'] = 0  
    _global_dict['numship1'] = 0  
    _global_dict['numtrue1'] = 0 

    _global_dict['numshiptrue2'] = 0  
    _global_dict['numship2'] = 0
    _global_dict['numtrue2'] = 0

    _global_dict['numshiptrue3'] = 0  
    _global_dict['numship3'] = 0 
    _global_dict['numtrue3'] = 0
     
    _global_dict['numshiptrue4'] = 0  
    _global_dict['numship4'] = 0
    _global_dict['numtrue4'] = 0

    _global_dict['picsum'] = 0         
    _global_dict['picship'] = 0       

    _global_dict['picshiptrue1'] = 0    
    _global_dict['picemptytrue1'] = 0  
    _global_dict['picselectasship1'] = 0     

    _global_dict['picshiptrue2'] = 0    
    _global_dict['picemptytrue2'] = 0 
    _global_dict['picselectasship2'] = 0         

    _global_dict['picshiptrue3'] = 0   
    _global_dict['picemptytrue3'] = 0 
    _global_dict['picselectasship3'] = 0       

    _global_dict['picshiptrue4'] = 0 
    _global_dict['picemptytrue4'] = 0 
    _global_dict['picselectasship4'] = 0  