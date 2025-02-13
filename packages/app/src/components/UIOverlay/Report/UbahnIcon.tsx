import { ComponentProps } from 'react'
import { Path, Svg } from 'react-native-svg'

import { FFView } from '../../common/base'

export const UbahnIcon = (props: ComponentProps<typeof FFView>) => (
    <FFView borderRadius="s" p="xxxs" style={{ backgroundColor: '#039' }} {...props}>
        <Svg enable-background="new 0 0 100 100" viewBox="0 0 100 100" width={30} height={30}>
            <Path d="M1 1h98v98H1z" fill="#039" />
            <Path
                d="M53.729 91.559c6.363-.621 12.307-2.645 17.028-5.799 2.372-1.585 5.718-4.866 7.172-7.034 2.126-3.169 3.669-6.976 4.495-11.086.32-1.592.359-4.39.424-30.38l.072-28.626H63.307v26.21c0 17.23-.076 26.888-.221 28.188-.26 2.329-1.026 5.114-1.804 6.555-.976 1.809-2.777 3.53-4.656 4.45-5.17 2.531-12.734 1.62-16.319-1.966-1.826-1.826-3.081-4.766-3.607-8.454-.129-.901-.208-11.505-.21-28.214l-.004-26.769H17.093v27.109c0 27.197.052 29.493.748 32.929 1.718 8.486 7.012 15.24 15.117 19.286 5.94 2.965 13.63 4.298 20.771 3.601z"
                fill="#fff"
            />
        </Svg>
    </FFView>
)
