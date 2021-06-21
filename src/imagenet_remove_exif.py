import glob
import piexif


if __name__ == '__main__':

    print("Training Dataset.")
    nfiles = 0
    for filename in glob.iglob('/srv/beegfs01/projects/imagenet/data/train/**/*.JPEG', recursive=True):
        nfiles = nfiles + 1
        print("About to process file %d, which is %s." % (nfiles, filename))
        piexif.remove(filename)
    print("%d files processed." % nfiles)

    print("Validation Dataset.")
    nfiles = 0
    for filename in glob.iglob('/srv/beegfs01/projects/imagenet/data/val/**/*.JPEG', recursive=True):
        nfiles = nfiles + 1
        print("About to process file %d, which is %s." % (nfiles, filename))
        piexif.remove(filename)
    print("%d files processed." % nfiles)

    print('Finish!')
