try:
    print("Workflow testing - dummy branch")
    raise Exception("Not found")
except:
    exit(1)