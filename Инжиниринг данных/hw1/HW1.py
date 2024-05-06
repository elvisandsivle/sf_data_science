import luigi
from luigi.util import requires

import pandas as pd

import tarfile
import gzip 
import wget
import os
import io

    
DEFAULT_PARAMS = {
    'dataset_name': 'GSE68849',
    'folder_name': 'datasets',
    'cols_to_delete': [
        'Definition', 
        'Ontology_Component', 
        'Ontology_Process', 
        'Ontology_Function', 
        'Synonyms', 
        'Obsolete_Probe_Id', 
        'Probe_Sequence'
    ]
}

def create_dir(folder_name: str) -> None:
    os.makedirs(folder_name, exist_ok=True)

class DownloadDatasetTask(luigi.Task):
    
    dataset_name = luigi.Parameter()
    folder_name = luigi.Parameter()

    def run(self):
        url = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={self.dataset_name}&format=file'
        
        create_dir(self.dir_name)
        output_path = self.output().path
        wget.download(url, out=output_path)
        
    def output(self):
        self.dir_name = f"{self.folder_name}/{self.dataset_name}"
        return luigi.LocalTarget(f'{self.dir_name}/{self.dataset_name}_RAW.tar')
        

@requires(DownloadDatasetTask)
class ExstractFromTarTask(luigi.Task):
    
    paths = []
       
    def run(self):
        self.dir, self.file = os.path.split(str(self.input()))
        os.chdir(self.dir)
        
        with tarfile.open(self.file, 'r') as tar:
            ind = 1
            
            for member in tar.getmembers():
                
                if member.name.endswith('.txt.gz'):
                    
                    folder_name = self.dataset_name + f'_{ind}'
                    create_dir(folder_name)
                    tar.extract(member, path=folder_name)
                    
                    path = os.path.abspath(f'{folder_name}/{member.name}')
                    self.paths.append(path)
                    ind += 1
    
    def output(self):
        return [luigi.LocalTarget(i) for i in self.paths] 
    
    
@requires(ExstractFromTarTask)
class ExstractFromGzipTask(luigi.Task):
    
    paths = []
    
    def run(self):
        
        for path in self.input():
            path = str(path)
            out_path = path.replace('.gz', '')
 
            with gzip.open(path, 'rb') as f_in, \
                  open(out_path, 'wb') as f_out:
                f_out.write(f_in.read())
            
            self.paths.append(out_path)
            os.remove(path)
    
    def output(self):
        return [luigi.LocalTarget(i) for i in self.paths] 
    
      
@requires(ExstractFromGzipTask)
class ExstractTsvsTask(luigi.Task):
    
    paths = []
    
    def run(self):
        for path in self.input():
            path = str(path)
            
            dir = os.path.split(path)[0]
            dfs = {}
            
            with open(path) as f:
                
                write_key = None
                fio = io.StringIO()
                
                for l in f.readlines():
                    
                    if l.startswith('['):
                        if write_key:
                            
                            fio.seek(0)
                            header = None if write_key == 'Heading' else 'infer'
                            dfs[write_key] = pd.read_csv(fio, sep='\t', header=header)
                            
                        fio = io.StringIO()
                        write_key = l.strip('[]\n')    
                        continue
                    
                    if write_key:
                        fio.write(l)
                        
                fio.seek(0)
                dfs[write_key] = pd.read_csv(fio, sep='\t')
                
            for name, df in dfs.items():
                
                path = os.path.join(dir, name+'.tsv')
                df.to_csv(path, sep='\t', encoding='utf-8')
                
                if name == 'Probes':
                    self.paths.append(path)                    
                
    def output(self):
        return [luigi.LocalTarget(i) for i in self.paths]
    
    
@requires(ExstractTsvsTask)
class DfCompressTask(luigi.Task):
    
    paths = []
    cols_to_delete = luigi.ListParameter()
    
    def run(self):
        
        for path in self.input():
            
            path = str(path)
            dir, file = os.path.split(path)
            file, exp = file.split('.')
            file = '.'.join([file+'_compressed', exp])
            out_path = os.path.join(dir, file)
            
            df = pd.read_csv(path, sep='\t')
            
            for col in self.cols_to_delete:
                
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            df.to_csv(out_path, sep='\t', encoding='utf-8')
            self.paths.append(out_path)
            
    def output(self):
        return [luigi.LocalTarget(i) for i in self.paths]
  
  
@requires(DfCompressTask)  
class RemoveTxtTask(luigi.Task):
    
    def run(self):
        for path in self.input():
            dir = os.path.split(str(path))[0]
            
            for file in os.listdir(dir):
                
                if file.endswith('.txt'):
                    os.remove(os.path.join(dir, file))


@requires(RemoveTxtTask)
class Process(luigi.WrapperTask):
     
    dataset_name = luigi.Parameter(default=DEFAULT_PARAMS['dataset_name'])
    folder_name = luigi.Parameter(default=DEFAULT_PARAMS['folder_name'])
    cols_to_delete = luigi.Parameter(default=DEFAULT_PARAMS['cols_to_delete'])
    
    def run(self):
        assert True
        

if __name__ == '__main__':
    luigi.build([Process()], workers=1, local_scheduler=True)