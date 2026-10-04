import re
from typing import Dict, List, Any

class CitationChecker:
    """
    Xác thực cấu trúc trích dẫn (Structural Validation) không dùng LLM.
    Chỉ kiểm tra tính hợp lệ của định dạng [CID=xxx] và tính nhất quán với allowed_cids.
    """

    @staticmethod
    def check(
        answer: str,
        declared_cids: List[Any],
        allowed_cids: List[Any],
        answer_status: str
    ) -> Dict[str, Any]:
        # Chuẩn hóa về string và deduplicate (giữ nguyên thứ tự)
        allowed = list(dict.fromkeys([str(c).strip() for c in allowed_cids if c is not None]))
        declared = list(dict.fromkeys([str(c).strip() for c in declared_cids if c is not None]))
        
        # SỬA LỖI Ở ĐÂY: Xử lý thông minh các trường hợp gộp CID
        if not answer:
            answer = ""
            
        raw_matches = re.findall(r'\[CID=(.*?)\]', answer, flags=re.IGNORECASE)
        extracted_cids = []
        
        for match in raw_matches:
            # match có thể là "123", "123, 456", hoặc "123, CID=456"
            # Tách bằng dấu phẩy hoặc từ khóa "CID="
            parts = re.split(r',|\bCID=', match, flags=re.IGNORECASE)
            for part in parts:
                # Xóa khoảng trắng và dấu "=" dư thừa nếu có
                clean_cid = re.sub(r'^=\s*', '', part.strip()).strip()
                if clean_cid:
                    extracted_cids.append(clean_cid)
                    
        inline = list(dict.fromkeys(extracted_cids))
        
        is_valid = True
        warnings = []
        
        # Phân tích sự bất đồng nhất (Inconsistencies)
        invalid_cids = []
        for c in inline + declared:
            if c not in allowed and c not in invalid_cids:
                invalid_cids.append(c)
                
        missing_declared_cids = [c for c in inline if c not in declared]
        uncited_declared_cids = [c for c in declared if c not in inline]
        
        has_any_citation = len(inline) > 0

        # Logic Validation
        if invalid_cids:
            is_valid = False
            warnings.append(f"Tồn tại CID bịa đặt ngoài context: {invalid_cids}")
            
        if answer_status == "answered" and not has_any_citation:
            is_valid = False
            warnings.append("Trạng thái 'answered' nhưng không có bất kỳ trích dẫn inline nào.")
            
        if missing_declared_cids:
            warnings.append(f"Có CID dùng inline nhưng không được khai báo trong cited_cids: {missing_declared_cids}")
            
        if uncited_declared_cids:
            warnings.append(f"Có CID khai báo trong cited_cids nhưng không được dùng inline: {uncited_declared_cids}")

        return {
            "is_valid": is_valid,
            "inline_cids": inline,
            "declared_cids": declared,
            "allowed_cids": allowed,
            "invalid_cids": invalid_cids,
            "missing_declared_cids": missing_declared_cids,
            "uncited_declared_cids": uncited_declared_cids,
            "has_any_citation": has_any_citation,
            "warnings": warnings
        } 