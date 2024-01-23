class NetworkManagerArea:
    def __init__(self,branch_id,contour,predicted):
        self.branch_id = branch_id
        self.contour = contour
        self.predicted = predicted

    def __str__(self) -> str:
        return "(branch_id : "+str(self.branch_id)+", contour : "+str(self.contour)+", predicted : "+str(self.predicted)+")"